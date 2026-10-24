#include "llama.h"
#include "chat.h"
#include "json.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

static std::vector<llama_token> tokens(const llama_vocab * vocab, const std::string & text, bool special) {
    int n = llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, special, true);
    if (n == 0) return {};
    if (n > 0) throw std::runtime_error("unexpected tokenizer sizing result");
    std::vector<llama_token> out(size_t(-n));
    n = llama_tokenize(vocab, text.data(), int(text.size()), out.data(), int(out.size()), special, true);
    if (n < 0) throw std::runtime_error("tokenization failed");
    out.resize(size_t(n));
    return out;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: prefix-vocab-probe model requests.json browser-results.json");
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], params), llama_model_free);
        if (!model) throw std::runtime_error("vocabulary load failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        auto tmpls = common_chat_templates_init(model.get(), "");
        std::ifstream request_file(argv[2]), result_file(argv[3]);
        json requests, results;
        request_file >> requests;
        result_file >> results;
        std::vector<std::string> prompts;
        std::vector<std::vector<llama_token>> encoded;
        json report = {{"vocab_only", true}, {"n_gpu_layers", 0}, {"contexts_created", 0},
            {"inference_executed", false}, {"turns", json::array()}, {"transitions", json::array()}};
        for (size_t i = 0; i < requests.size(); ++i) {
            common_chat_templates_inputs input;
            input.messages = common_chat_msgs_parse_oaicompat(common_json::parse(requests[i]["messages"].dump()));
            input.enable_thinking = true;
            input.add_generation_prompt = true;
            input.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            auto formatted = common_chat_templates_apply(tmpls.get(), input);
            prompts.push_back(formatted.prompt);
            encoded.push_back(tokens(vocab, formatted.prompt, true));
            const size_t recorded = results[i]["usage"]["prompt_tokens"];
            report["turns"].push_back({{"turn", i+1}, {"rendered_prompt_tokens", encoded.back().size()},
                {"recorded_prompt_tokens", recorded}, {"count_matches", encoded.back().size() == recorded}});
            if (i == 0) continue;
            const auto & previous = prompts[i-1];
            const bool text_extends = prompts[i].compare(0, previous.size(), previous) == 0;
            size_t token_prefix = 0;
            while (token_prefix < std::min(encoded[i-1].size(), encoded[i].size()) &&
                    encoded[i-1][token_prefix] == encoded[i][token_prefix]) ++token_prefix;
            const auto user = prompts[i].rfind("<|im_start|>user\n");
            const std::string close = "<|im_end|>\n";
            json transition = {{"from_turn", i}, {"to_turn", i+1}, {"previous_input_text_is_exact_prefix", text_extends},
                {"previous_input_token_prefix", token_prefix}, {"recorded_cache_hit", results[i]["usage"]["prompt_cache_hit_tokens"]}};
            if (text_extends && user != std::string::npos && user >= previous.size()+close.size() &&
                    prompts[i].compare(user-close.size(), close.size(), close) == 0) {
                auto canonical_generation = prompts[i].substr(previous.size(), user-close.size()-previous.size());
                auto raw_api_reconstruction = results[i-1].value("reasoning_content", std::string{}) +
                    "</think>\n\n" + results[i-1].value("content", std::string{});
                auto whole = tokens(vocab, previous+canonical_generation, true);
                const size_t actual_generation = results[i-1]["usage"]["completion_tokens"];
                transition["recorded_generation_tokens"] = actual_generation;
                transition["canonical_generation_tokens_standalone"] = tokens(vocab, canonical_generation, false).size();
                transition["canonical_prefix_tokens_without_turn_close"] = whole.size();
                transition["recorded_cached_token_count"] = encoded[i-1].size()+actual_generation;
                transition["api_reconstructed_text_equals_canonical"] = raw_api_reconstruction == canonical_generation;
                transition["canonical_generation_bytes"] = canonical_generation.size();
                transition["api_reconstruction_bytes"] = raw_api_reconstruction.size();
            }
            report["transitions"].push_back(std::move(transition));
        }
        std::cout << report.dump(2) << '\n';
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

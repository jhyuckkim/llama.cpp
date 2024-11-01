#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <set>

// Map a slim ID to the corresponding original token ID
llama_token slim_id_to_token_id(llama_token slim_id, const std::vector<llama_token>& sorted_list) {
    if (slim_id >= 0 && slim_id < sorted_list.size()) {
        return sorted_list[slim_id];
    } else {
        std::cerr << "slim_id out of range!" << std::endl;
        return -1;
    }
}

// Map an original token ID to the corresponding slim ID
llama_token token_id_to_slim_id(llama_token token_id, const std::unordered_map<llama_token, llama_token>& mapping) {
    if (mapping.find(token_id) != mapping.end()) {
        return mapping.at(token_id);
    } else {
        std::cerr << "token_id not found in the sorted list!" << std::endl;
        return -1;
    }
}

// Load the slimmed vocabulary from file
std::vector<int> get_slimed_vocab(const std::string& model_path) {
    std::filesystem::path slimed_vocab_path = std::filesystem::path(model_path).parent_path() / "slimed_vocab.txt";
    std::vector<int> slimed_vocab;
    std::ifstream slimed_vocab_file(slimed_vocab_path);
    if (!slimed_vocab_file) {
        std::cerr << "Error: Could not open slimed_vocab.txt at " << slimed_vocab_path << std::endl;
        return slimed_vocab;
    }

    int number;
    while (slimed_vocab_file >> number) {
        slimed_vocab.push_back(number);
    }
    return slimed_vocab;
}

// Print usage information for command-line arguments
static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "Artificial intelligence is transforming";
    int ngl = 99;
    int n_predict = 32;

    // Parse command-line arguments
    int i = 1;
    for (; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            ngl = std::stoi(argv[++i]);
        } else {
            break;
        }
    }
    if (model_path.empty()) {
        print_usage(argc, argv);
        return 1;
    }
    if (i < argc) {
        prompt = argv[i++];
        for (; i < argc; i++) prompt += " " + std::string(argv[i]);
    }

    // Initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // Tokenize the prompt
    const int n_prompt = -llama_tokenize(model, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(model, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // Print initial prompt tokens
    std::cout << "Initial prompt tokens: ";
    for (const llama_token& token : prompt_tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    // Initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // Initialize the sampler
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Load slimmed vocabulary and map prompt tokens to slim IDs
    std::vector<llama_token> sorted_subset_ids = get_slimed_vocab(model_path);
    std::unordered_map<llama_token, llama_token> token_to_slim_map;
    for (llama_token i = 0; i < sorted_subset_ids.size(); i++) {
        token_to_slim_map[sorted_subset_ids[i]] = i;
    }
    for (llama_token i = 0; i < prompt_tokens.size(); i++) {
        prompt_tokens[i] = token_id_to_slim_id(prompt_tokens[i], token_to_slim_map);
    }

    // Prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // Generate text
    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id, new_slim_token_id;
    std::string generated_text = prompt;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
        n_pos += batch.n_tokens;

        // Sample the next token
        new_slim_token_id = llama_sampler_sample(smpl, ctx, -1);
        new_token_id = slim_id_to_token_id(new_slim_token_id, sorted_subset_ids);

        // Check for end of generation
        if (llama_token_is_eog(model, new_token_id)) break;

        // Convert token to text and append
        char buf[128];
        int n = llama_token_to_piece(model, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::cout << "new_slim_token_id: " << new_slim_token_id 
                  << " -> original_token_id: " << new_token_id 
                  << " -> text: " << std::string(buf, n) << std::endl;
        generated_text += std::string(buf, n);

        // Prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_slim_token_id, 1);
        n_decode += 1;
    }

    printf("\nGenerated text:\n%s\n", generated_text.c_str());

    const auto t_main_end = ggml_time_us();
    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));
    
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_free_model(model);

    return 0;
}


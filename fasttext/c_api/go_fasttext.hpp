#ifndef GO_FASTTEXT_HPP
#define GO_FASTTEXT_HPP

#ifdef __cplusplus 
extern "C" {
#endif

typedef struct {
    char* label;
    float prob;                                                                                         
} go_fast_text_pair_t;

int ft_load_model(const char* path);

go_fast_text_pair_t* ft_predict(const char *query_in, int k, float threshold, int* result_length);

int ft_get_vector_dimension();

int ft_get_sentence_vector(const char* query_in, float* vector, int vector_size);

int train(const char* model_name, const char* input, const char* output, int epoch, int word_ngrams, int thread, float lr);

int quantize(const char* input, const char* output);

int ft_save_model(const char* filename);

int ft_delete();

#ifdef __cplusplus
}
#endif

#endif
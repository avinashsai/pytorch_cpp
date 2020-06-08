#include<torch/torch.h>
#include<torch/script.h>
#include<cstdio>
#include<iostream>
#include<float.h>
#include<fstream>
#include<sstream>
#include<string>
#include<regex>
#include<tuple>
#include<typeinfo>
#include<vector>
#define seed 0
#include "loader.h"
#include "datautils.h"
#include "model.h"
#include "trainmodel.h"

// Path to data
std::string datapath = "../../Data/TREC/";

// Path to pretrained vectors
std::string vector_path = "../../glove.840B.300d.txt";

// Maximum length of sentence
int32_t kMaxLen = 20;

// Embedding Dimensions
const int32_t kEmbedDim  = 300;

// Number of classes in TREC Dataset
int32_t kNumClasses = 6;

// Number of filters
const int32_t kNumfilters = 100;

// Filter sizes
int32_t kFilterSizes[] = {3, 4, 5};

// Dropout value
double kDropValue = 0.5;

// Batch size
int32_t kBatchSize =  16;

// Number of epochs to train
int32_t kNumofEpochs = 10;

int main(int argc, char** argv) {
    torch::manual_seed(seed);

    std::string model_name;

    model_name = argv[1];

    std::string corpus;
    std::vector<int32_t> labels;
    std::vector<std::string> train_corpus;
    std::vector<int32_t> train_labels;

    std::vector<std::string> test_corpus;
    std::vector<int32_t> test_labels;

    auto ktrain = load_trec(datapath, "train.txt");
    std::cout << "Loaded Training data" << std::endl;
    train_corpus = ktrain.first;
    train_labels = ktrain.second;

    auto ktest = load_trec(datapath, "test.txt");
    std::cout << "Loaded Testing data" << std::endl;
    test_corpus = ktest.first;
    test_labels = ktest.second;

    if (train_corpus.size() == train_labels.size()) {
        std::cout << "Training Data Size:" << " " << train_corpus.size();
        std:: cout << std::endl;
    }

    if (test_corpus.size() == test_labels.size()) {
        std::cout << "Test Data Size:" << " " << test_corpus.size();
        std::cout << std::endl;
    }

    std::map<std::string, int32_t> vocab;
    vocab = create_vocab(train_corpus);

    std::pair<torch::Tensor, torch::Tensor> trainindices_labels;
    std::pair<torch::Tensor, torch::Tensor> testindices_labels;

    // Converting Training Data into Training Data Loader
    trainindices_labels = get_loader(vocab, train_corpus, 
        train_labels, kMaxLen, kBatchSize);

    auto traindata_set = CustomDataset(trainindices_labels.first,
        trainindices_labels.second).map(
             torch::data::transforms::Stack<>());

    auto train_dataset_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(traindata_set), kBatchSize);

    // Converting Testing Data into Test Data Loader
    testindices_labels = get_loader(vocab, test_corpus, test_labels,
        kMaxLen, kBatchSize);

    auto testdata_set = CustomDataset(testindices_labels.first,
        testindices_labels.second).map(
            torch::data::transforms::Stack<>());

    auto test_dataset_loader =
         torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(testdata_set), kBatchSize);

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "GPU available. Training on GPU." << std::endl;
        torch::Device device(torch::kCUDA);
    } else {
        std::cout << "No GPU available. Training on CPU. " << std::endl;
    }

    if (model_name == "random") {
        int32_t vocabsize = vocab.size();
        RandCNN randcnn(vocabsize, kEmbedDim, kMaxLen, kNumClasses, kNumfilters,
            kFilterSizes, kDropValue);
        randcnn.to(device);

        train_model(
            kNumofEpochs,
            randcnn,
            *train_dataset_loader,
            *test_dataset_loader,
            device);
    } else if (model_name == "static" || model_name == "nonstatic") {
        auto embed_matrix = get_embedding_matrix(
                                                vocab, 
                                                vector_path, 
                                                kEmbedDim);
        
        bool dotrain = false;
        if (model_name == "nonstatic"){
            dotrain = true;
            kNumofEpochs = 25;
        }

        CNN cnn(embed_matrix, kEmbedDim, kMaxLen, kNumClasses, kNumfilters,
            kFilterSizes, kDropValue, dotrain);
        cnn.to(device);

        train_model(
            kNumofEpochs,
            cnn,
            *train_dataset_loader,
            *test_dataset_loader,
            device);
            
    } else {
        auto embed_matrix = get_embedding_matrix(
                                                vocab, 
                                                vector_path, 
                                                kEmbedDim);

        MultiCNN multicnn(embed_matrix, kEmbedDim, kMaxLen, kNumClasses, 
            kNumfilters, kFilterSizes, kDropValue);
        multicnn.to(device);

        train_model(
            kNumofEpochs,
            multicnn,
            *train_dataset_loader,
            *test_dataset_loader,
            device);
    }
}

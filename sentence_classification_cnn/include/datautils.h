#ifndef datautils
#define datautils

#include<torch/torch.h>
#include<cstdio>
#include<iostream>
#include<string>
#include<vector>
typedef std::map<std::vector<std::string>,
std::vector<int32_t> > innermap;

// Function to load pretrained word vectors and converts words to
// embedding matrix.
torch::Tensor get_embedding_matrix(
    std::map<std::string, int32_t>, 
    std::string,
    int32_t);

// Function to split data into train and test
std::pair<innermap, innermap> splitdata(
    std::vector<std::string> &sentences,
    std::vector<int32_t> &labels,
    float test_size);

// Function to load TREC dataset
std::pair< std::vector<std::string>, std::vector<int32_t> > load_trec(
    std::string datapath,
    std::string name);

// Function to load MR and SO dataset
std::pair< std::vector<std::string>, std::vector<int32_t> > load_all(
    std::string datapath,
    std::string file1,
    std::string file2);

// Function to create vocabulary
std::map<std::string, int32_t> create_vocab(std::vector<std::string>);

// Function to return tensor of indices for a sentence with padding
torch::Tensor get_indices(
                        std::string,
                        std::map<std::string,
                        int32_t>,
                        int32_t);

// Function to return dataloader with sentences converted to
// vocabulary indices and labels converted to tensors
std::pair<torch::Tensor, torch::Tensor> get_loader(
    std::map<std::string, int32_t> &dic,
    std::vector<std::string> &sen,
    std::vector<int32_t> &labels,
    int32_t,
    int32_t);

// Class to convert indices, labels into DataLoader
class CustomDataset : public torch::data::Dataset<CustomDataset> {
 private:
        torch::Tensor inputs_, targets_;
 public:
        CustomDataset(torch::Tensor emb_indices, torch::Tensor labels) {
            inputs_ = emb_indices;
            targets_ = labels;
         }
         torch::data::Example<> get(size_t index) override {
            torch::Tensor embedding_index = inputs_[index];
            torch::Tensor Label = targets_[index];
            return {embedding_index.clone(), Label.clone()};
         };
         torch::optional<size_t> size() const override{
            return at::size(inputs_, 0);
         };
};
#endif  /* datautils.h */

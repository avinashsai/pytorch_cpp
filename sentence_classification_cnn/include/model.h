#ifndef model
#define model

#include<torch/torch.h>

using namespace torch;
using namespace at;

struct RandCNN : nn::Module {
  RandCNN(int32_t vocabsize, int32_t embeddim, int32_t maxlen,
        int32_t numclasses, int32_t kNumFilters, int32_t kFilterSizes[],
        double kDropValue)
      : embed(nn::Embedding(nn::EmbeddingOptions(vocabsize, 
                                                embeddim))),
        conv1(nn::Conv2d(nn::Conv2dOptions(1, kNumFilters, {kFilterSizes[0], embeddim})
                            .stride(1)
                            .bias(false))),
        conv2(nn::Conv2d(nn::Conv2dOptions(1, kNumFilters, {kFilterSizes[1], embeddim})
                            .stride(1)
                            .bias(false))),
        conv3(nn::Conv2d(nn::Conv2dOptions(1, kNumFilters, {kFilterSizes[2], embeddim})
                            .stride(1)
                            .bias(false))),
        pool1(nn::MaxPool1d(nn::MaxPool1dOptions(maxlen - kFilterSizes[0] + 1)
                               .stride(1))),
        pool2(nn::MaxPool1d(nn::MaxPool1dOptions(maxlen - kFilterSizes[1] + 1)
                               .stride(1))),
        pool3(nn::MaxPool1d(nn::MaxPool1dOptions(maxlen - kFilterSizes[2] + 1)
                               .stride(1))),
        fc(nn::Linear(nn::LinearOptions(kNumFilters * 3, numclasses))),
        drop(nn::Dropout(nn::DropoutOptions().p(kDropValue))) {

   // register_module() is needed if we want to use the parameters() method later on
    register_module("embed", embed);
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("poo11", pool1);
    register_module("pool2", pool2);
    register_module("pool3", pool3);
    register_module("fc", fc);
}

 torch::Tensor forward(torch::Tensor x) {
    x = embed(x);
    x = at::unsqueeze(x, 1);
    torch::Tensor out1, out2, out3, out;
    out1 = torch::relu(conv1->forward(x)).squeeze(3);
    out2 = torch::relu(conv2->forward(x)).squeeze(3);
    out3 = torch::relu(conv3->forward(x)).squeeze(3);
    out1 = pool1(out1);
    out2 = pool2(out2);
    out3 = pool3(out3);
    out = at::cat({out1, out2, out3}, 1);
    out = at::_unsafe_view(out, {at::size(out, 0), at::size(out, 1)});
    out = drop(out);
    out = fc(out);
   return out;
 }

 nn::Embedding embed;
 nn::Conv2d conv1, conv2, conv3;
 nn::MaxPool1d pool1, pool2, pool3;
 nn::Linear fc;
 nn::Dropout drop;
};

struct CNN : nn::Module {
    CNN(torch::Tensor &embed_matrix, int32_t embeddim, int32_t maxlen,
        int32_t numclasses, int32_t kNumFilters, int32_t kFilterSizes[], 
        double kDropValue, bool &do_train_embed)
      : embed(nn::Embedding::from_pretrained(embed_matrix,
                        nn::EmbeddingFromPretrainedOptions().freeze(do_train_embed))),
        conv1(nn::Conv2d(nn::Conv2dOptions(1, kNumFilters, {kFilterSizes[0], embeddim})
                            .stride(1)
                            .bias(false))),
        conv2(nn::Conv2d(nn::Conv2dOptions(1, kNumFilters, {kFilterSizes[1], embeddim})
                            .stride(1)
                            .bias(false))),
        conv3(nn::Conv2d(nn::Conv2dOptions(1, kNumFilters, {kFilterSizes[2], embeddim})
                            .stride(1)
                            .bias(false))),
        pool1(nn::MaxPool1d(nn::MaxPool1dOptions(maxlen - kFilterSizes[0] + 1)
                               .stride(1))),
        pool2(nn::MaxPool1d(nn::MaxPool1dOptions(maxlen - kFilterSizes[1] + 1)
                               .stride(1))),
        pool3(nn::MaxPool1d(nn::MaxPool1dOptions(maxlen - kFilterSizes[2] + 1)
                               .stride(1))),
        fc(nn::Linear(nn::LinearOptions(kNumFilters * 3, numclasses))),
        drop(nn::Dropout(nn::DropoutOptions().p(kDropValue))) {

   // register_module() is needed if we want to use the parameters() method later on
    register_module("embed", embed);
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("poo11", pool1);
    register_module("pool2", pool2);
    register_module("pool3", pool3);
    register_module("fc", fc);
}

 torch::Tensor forward(torch::Tensor x) {
    x = embed(x);
    //std::cout<<x[0][0][0]<<"\n";
    x = x.unsqueeze(1);
    torch::Tensor out1, out2, out3, out;
    out1 = torch::relu(conv1->forward(x)).squeeze(3);
    out2 = torch::relu(conv2->forward(x)).squeeze(3);
    out3 = torch::relu(conv3->forward(x)).squeeze(3);
    out1 = pool1(out1);
    out2 = pool2(out2);
    out3 = pool3(out3);
    out = at::cat({out1, out2, out3}, 1);
    out = at::_unsafe_view(out, {at::size(out, 0), at::size(out, 1)});
    out = drop(out);
    out = fc(out);
   return out;
 }

 nn::Embedding embed;
 nn::Conv2d conv1, conv2, conv3;
 nn::MaxPool1d pool1, pool2, pool3;
 nn::Linear fc;
 nn::Dropout drop;
};

struct MultiCNN : nn::Module {
    MultiCNN(torch::Tensor& embed_matrix, int32_t embeddim, int32_t maxlen,
        int32_t numclasses, int32_t kNumFilters, int32_t kFilterSizes[], 
        double kDropValue)
    :   embed1(nn::Embedding::from_pretrained(embed_matrix,
                        nn::EmbeddingFromPretrainedOptions().freeze(true).padding_idx(0))),
        embed2(nn::Embedding::from_pretrained(embed_matrix,
                        nn::EmbeddingFromPretrainedOptions().freeze(false))),
        conv1(nn::Conv2d(nn::Conv2dOptions(1, kNumFilters, {kFilterSizes[0], embeddim * 2})
                            .stride(1)
                            .bias(false))),
        conv2(nn::Conv2d(nn::Conv2dOptions(1, kNumFilters, {kFilterSizes[1], embeddim * 2})
                            .stride(1)
                            .bias(false))),
        conv3(nn::Conv2d(nn::Conv2dOptions(1, kNumFilters, {kFilterSizes[2], embeddim * 2})
                            .stride(1)
                            .bias(false))),
        pool1(nn::MaxPool1d(nn::MaxPool1dOptions(maxlen - kFilterSizes[0] + 1)
                               .stride(1))),
        pool2(nn::MaxPool1d(nn::MaxPool1dOptions(maxlen - kFilterSizes[1] + 1)
                               .stride(1))),
        pool3(nn::MaxPool1d(nn::MaxPool1dOptions(maxlen - kFilterSizes[2] + 1)
                               .stride(1))),
        fc(nn::Linear(nn::LinearOptions(kNumFilters * 3, numclasses))),
        drop(nn::Dropout(nn::DropoutOptions().p(kDropValue))) {

   // register_module() is needed if we want to use the parameters() method later on
    register_module("embed1", embed1);
    register_module("embed2", embed2);
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("poo11", pool1);
    register_module("pool2", pool2);
    register_module("pool3", pool3);
    register_module("fc", fc);
}

 torch::Tensor forward(torch::Tensor x) {
    torch::Tensor x1, x2;
    torch::Tensor out1, out2, out3, out;
    x1 = embed1(x);
    x2 = embed2(x);
    x = at::cat({x1, x2}, 2).unsqueeze(1);
    out1 = torch::relu(conv1->forward(x)).squeeze(3);
    out2 = torch::relu(conv2->forward(x)).squeeze(3);
    out3 = torch::relu(conv3->forward(x)).squeeze(3);
    out1 = pool1(out1);
    out2 = pool2(out2);
    out3 = pool3(out3);
    out = at::cat({out1, out2, out3}, 1);
    out = at::_unsafe_view(out, {at::size(out, 0), at::size(out, 1)});
    out = drop(out);
    out = fc(out);
   return out;
 }

 nn::Embedding embed1, embed2;
 nn::Conv2d conv1, conv2, conv3;
 nn::MaxPool1d pool1, pool2, pool3;
 nn::Linear fc;
 nn::Dropout drop;
};
#endif  /* model.h */

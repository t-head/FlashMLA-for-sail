/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/
#pragma once
#include <hgtx3/hgToolsExt.h>
#include <string>
#include <iostream>

namespace ppu {
namespace fmha {

class FmhaProfParam {
public:

FmhaProfParam() {}

void initialize_args() {
  add_argument("data_type");     // string
  add_argument("batch_size");    // int
  add_argument("num_heads");
  add_argument("num_heads_k");
  add_argument("head_dim");
  add_argument("head_dim_value");
  add_argument("seqlen_q");
  add_argument("seqlen_k");
  add_argument("custom_mask");   //bool
  // add_argument("sparse");
  // add_argument("topk");
  // add_argument("is_fp8");
}

template <typename T>
void add_mha_params(const std::string& key, const T& val) {
  if (args_.find(key) == args_.end()) {
    args_.insert(std::make_pair(key, val_to_string(val)));
  }
  args_.at(key) = val_to_string(val);
  insertionOrder.push_back(key);
}

template <typename T>
std::string nvtx_param2str(void* addr, int sz, hggcStream_t stream) {
  std::vector<T> tmp(sz);
  hggcMemcpyAsync(tmp.data(), addr, sizeof(T) * sz, hggcMemcpyDeviceToHost, stream);
  hggcStreamSynchronize(stream);

  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < sz; ++i) {
      oss << tmp[i];
      if (i < sz - 1) oss << ",";
  }
  oss << "]";
  return oss.str();
}

void set_flash_attn_params(bool is_bf16,
                           bool is_causal, int batch_size,
                           int num_heads, int num_heads_k,
                           int head_dim, int head_dim_value,
                           int seqlen_q, std::string seqlen_k,
                           int topk = -1, bool is_fp8 = false,
                           bool have_attn_sink = false,
                           std::string topk_len = "",
                           int extra_topk = -1,
                           std::string extra_topk_len = ""){

  initialize_args();
  bool is_sparse = topk > -1;
  std::string data_type = is_bf16 ? "bf16" : "fp16";
  if (is_sparse) {
    std::string sparse_type = "decode";
    add_mha_params("sparse", sparse_type);
    add_mha_params("is_fp8", is_fp8);
    add_mha_params("topk", topk);
    add_mha_params("have_attn_sink", have_attn_sink);
    if (!topk_len.empty()) {
      add_mha_params("topk_len", topk_len);
    }
    if (extra_topk > -1) {
      add_mha_params("extra_topk", extra_topk);
      if (!extra_topk_len.empty()) {
        add_mha_params("extra_topk_len", extra_topk_len);
      }
    }
  }

  add_mha_params("batch_size", batch_size);
  add_mha_params("seqlen_q", seqlen_q);
  add_mha_params("seqlen_k", seqlen_k);
  add_mha_params("num_heads", num_heads);
  add_mha_params("num_heads_kv", num_heads_k);
  add_mha_params("head_dim", head_dim);
  add_mha_params("head_dim_v", head_dim_value);
  add_mha_params("causal", is_causal);
  add_mha_params("dtype", data_type);
}

void set_flash_attn_sparse_prefill_params(bool is_bf16,
                                  // bool is_causal, int batch_size,
                                  int num_heads, int num_heads_k,
                                  int head_dim, int head_dim_value,
                                  int seqlen_q, int seqlen_k, int topk,
                                  bool have_attn_sink = false,
                                  std::string topk_len = "") {

  initialize_args();
  std::string data_type = is_bf16 ? "bf16" : "fp16";

  // add_mha_params("batch_size", batch_size);
  std::string sparse_type = "prefill";
  add_mha_params("sparse", sparse_type);
  add_mha_params("topk", topk);
  add_mha_params("seqlen_q", seqlen_q);
  add_mha_params("seqlen_k", seqlen_k);
  add_mha_params("num_heads", num_heads);
  add_mha_params("num_heads_kv", num_heads_k);
  add_mha_params("head_dim", head_dim);
  add_mha_params("head_dim_v", head_dim_value);
  // add_mha_params("causal", is_causal);
  add_mha_params("dtype", data_type);
  add_mha_params("have_attn_sink", have_attn_sink);
  if (!topk_len.empty()) {
    add_mha_params("topk_len", topk_len);
  }
}

std::string format() {
  std::stringstream ss;

  ss << "[MLA] --format=";
  // for(auto& iter: args_) {
  //   if (iter.second != ""){
  //     ss << iter.first << ':' << iter.second << ',';
  //   }
  // }
  for (auto& key : insertionOrder) {
      ss << key << ":" << args_[key];
      if (&key != &insertionOrder.back())
        ss << ',';
  }
  ss << '.';
  return ss.str();
}

private:

std::string val_to_string(int val) {
  return std::to_string(val);
}

std::string val_to_string(float val) {
  std::stringstream float_str;
  float_str << std::fixed << std::setprecision(4) << val;
  return float_str.str();
}

std::string val_to_string(bool val) {
  return std::to_string(int(val));
}

std::string val_to_string(const std::string& val) {
  return val;
}

void add_argument(const std::string& name) {
  std::string init_val = "";
  if (args_.find(name) != args_.end()) {
    std::cout << "[" << name << "] already exists." << std::endl;
    throw std::runtime_error("Add argument fail.");
  }

  args_.insert(std::make_pair(name, init_val));
}

protected:
  std::unordered_map<std::string, std::string> args_;
  std::vector<std::string> insertionOrder;
};


class ProfilingInterface {
public:
  ProfilingInterface(ProfilingInterface const&) = delete;
  void operator=(ProfilingInterface const&) = delete;

  static ProfilingInterface& Instance() {
    static ProfilingInterface instance;
    return instance;
  }

  bool get_op_info() {
    return show_params_ || use_nvtx_;
  }

  void instrument(bool start, FmhaProfParam &fmha_params) {
    if (!get_op_info()){
      return;
    }

    if (start) {
      std::string op_name = fmha_params.format();
      if (show_params_) {
        std::cout << op_name << std::endl;
      }
      if (use_nvtx_) {
        hgtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = HGTX_VERSION;
        eventAttrib.messageType = HGTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = op_name.c_str();
        hgtxDomainRangePushEx(domain_, &eventAttrib);
      }
    } else {
      if (use_nvtx_) {
        hgtxDomainRangePop(domain_);
      }
    } // if start

  }

private:
  ProfilingInterface() {
    // TODO: add print log
    domain_ = hgtxDomainCreateA("mla");
    use_nvtx_ = false;
    show_params_ = false;

    char *pEnv_perf = std::getenv("PPU_LIB_PERF_INSTRUMENT");
    if (pEnv_perf && isdigit(*pEnv_perf)) {
      int value = std::stoi(std::string(pEnv_perf));
      if (value == 0) {
        use_nvtx_ = false;
      } else if (value == 1) {
        use_nvtx_ = true;
      } else {
        printf("Invalid value for PPU_LIB_PERF_INSTRUMENT : %d\n", value);
      }
    }

    char *pEnv_params = std::getenv("PPU_LIB_SHOW_PARAMS");
    if (pEnv_params && isdigit(*pEnv_params)) {
      int value = std::stoi(std::string(pEnv_params));
      if (value == 0) {
        show_params_ = false;
      } else if (value == 1) {
        show_params_ = true;
      } else {
        printf("Invalid value for PPU_LIB_SHOW_PARAMS : %d\n", value);
      }
    }
  }

  ~ProfilingInterface() {
    hgtxDomainDestroy(domain_);
  }

  bool use_nvtx_;
  bool show_params_;
  hgtxDomainHandle_t domain_;

};

} // namespace fmha
} // namespace ppu
//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>

void load_result_data(char* filename, std::vector<std::vector<unsigned>>& data, unsigned& num, unsigned& dim) { 
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        return;
    }
    in.read((char*)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    unsigned fsize = (unsigned)ss;
    num = fsize / (dim + 1) / 4;
    data.resize(num);

    in.seekg(0, std::ios::beg);
    for (unsigned i = 0; i < num; i++) {
        std::vector<unsigned> tmp(dim);
        in.seekg(4, std::ios::cur);
        in.read((char*)tmp.data(), dim * 4);
        data[i] = tmp;
    }
    in.close();
}

void eval_recall(std::vector<std::vector<unsigned>>& res, std::vector<std::vector<unsigned> > &gt, unsigned K){
  float mean_acc=0;
  for(unsigned i=0; i<res.size(); i++){
    float acc = 0;
    auto &g = res[i];
    auto &v = gt[i];
    for(unsigned j=0; j<K; j++){
      for(unsigned k=0; k<K; k++){
        if(g[j] == v[k]){
          acc++;
          break;
        }
      }
    }
    mean_acc += acc / K;
  }
  std::cout<<"RECALL@" << K << ": " << mean_acc / res.size() <<std::endl;
}

void load_bvecs(char* filename, uint8_t*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  std::cout << "data dimension: " << dim << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (4 + dim));
  data = new uint8_t[num * dim * sizeof(uint8_t)];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim);
  }
  in.close();
}

void load_fvecs(char* filename, float*& data, unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[num * dim * sizeof(float)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
}

void save_result(char* filename, std::vector<std::vector<unsigned> >& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

// ./test /home/zjlab/ANNS/yq/baseline/_tmp/_code/sift/pq/pq8x8/sift_codes.bvecs /home/zjlab/ANNS/yq/baseline/_tmp/_code/sift/pq/pq8x8/sift_lut.fvecs /home/zjlab/ANNS/yq/baseline/_tmp/_graph/sift_nsg.graph 512 10 ./ /home/zjlab/ANNS/yq/dataset/origin/sift/sift_groundtruth.ivecs
// ./test /home/zjlab/ANNS/yq/baseline/_tmp/_code/bigann1M/pq/pq8x8/bigann1M_codes.bvecs /home/zjlab/ANNS/yq/baseline/_tmp/_code/bigann1M/pq/pq8x8/bigann1M_lut.fvecs /home/zjlab/ANNS/yq/baseline/_tmp/_graph/bigann_nsg.graph 512 10 ./ /home/zjlab/ANNS/yq/dataset/origin/bigann_soft/gnd/idx_1M.ivecs
int main(int argc, char** argv) {
  if (argc != 8) {
    std::cout << argv[0]
              << " data_file query_file nsg_path search_L search_K result_path gt_path"
              << std::endl;
    exit(-1);
  }
  uint8_t* data_load = NULL;
  unsigned points_num, dim;
  load_bvecs(argv[1], data_load, points_num, dim);
  float* dist_load = NULL;
  unsigned query_num, chunk_num, cen_num = 256, total_num;
  load_fvecs(argv[2], dist_load, query_num, total_num);
  chunk_num = total_num / cen_num;
  std::vector<std::vector<unsigned>> gt_load;
  unsigned gt_num, gt_dim;
  load_result_data(argv[7], gt_load, gt_num, gt_dim);
  assert(dim == chunk_num);

  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build query_load = efanna2e::data_align(query_load,
  // query_num, query_dim);
  efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
  index.Load(argv[3]);

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);

  auto s = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<unsigned> > res(query_num, std::vector<unsigned>(K));
  unsigned *tmp = new unsigned[query_num * K];
  
  index.BatchSearchWithCodes(dist_load, query_num, chunk_num, cen_num, data_load, K, paras, tmp);

  for(int i = 0; i < query_num; i ++) {
    for(int j = 0; j < K; j ++) {
      // std::cout << tmp[i*K+j] << std::endl;
      res[i][j] = tmp[i * K + j];
    }
  }


  // for (unsigned i = 0; i < query_num; i++) {
  //   std::vector<unsigned> tmp(K);
  //   std::vector<unsigned> heap_his(L * 2 * L);
  //   std::memset(heap_his.data(), 0, sizeof(unsigned));
  //   // index.SearchWithCodes(dist_load + i * (chunk_num * cen_num), data_load, K, paras, tmp.data());
  //   // index.SearchCodesWithHeapInfo(dist_load + i * (chunk_num * cen_num), data_load, K, paras, tmp.data(), heap_his.data());
  //   // for(int i = 0; i < 10; i ++) {
  //   //   for(int j = 0; j < L; j ++) {
  //   //     std::cout << heap_his[i * L + j] << " ";
  //   //   }
  //   // }
  //   // std::cout << std::endl;
  //   res.push_back(tmp);
  // }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "search time: " << diff.count() << "\n";

  // std::cout << res.size() << " " << res[0].size() << std::endl;
  // for(int i = 0; i < res.size(); i ++) {
  //    for(int j = 0; j < res[0].size(); j++) {
  //      std::cout << res[i][j] << " ";
  //    }
  //    std::cout << std::endl;
  // }

  eval_recall(res, gt_load, K);
  save_result(argv[6], res);

  return 0;
}

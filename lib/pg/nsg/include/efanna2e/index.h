//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_INDEX_H
#define EFANNA2E_INDEX_H

#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include "distance.h"
#include "parameters.h"

namespace efanna2e {

class Index {
 public:
  explicit Index(const size_t dimension, const size_t n, Metric metric);


  virtual ~Index();

  virtual void Build(size_t n, const float *data, const Parameters &parameters) = 0;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) = 0;

  virtual void Save(const char *filename) = 0;

  virtual void Load(const char *filename) = 0;

  inline bool HasBuilt() const { return has_built; }

  inline void SetDataset(float *data) { data_ = data; }

  inline const float *GetDataset() const { return data_; }

  inline const void InitDistCount() { dist_count = 0; }

  inline const size_t GetDistCount() { return dist_count; }
  
 protected:
  const size_t dimension_;
  const float *data_;
  const uint8_t *code_;
  size_t nd_;
  bool has_built;
  Distance* distance_;
  size_t dist_count = 0;
  size_t hop_count = 0;
};

}

#endif //EFANNA2E_INDEX_H

#pragma once

#include <torch/all.h>

#include "StdCommon.h"
#include "Logging.h"
//==============================================================================
namespace sarathi
{
class RMSNorm
{
public:
    RMSNorm(
        const torch::Tensor& weight,
        double fVarianceEpsilon
    );

    torch::Tensor Forward(const torch::Tensor& input /*[in]*/) const;

private:
    const torch::Tensor m_weight;
    double m_fVarianceEpsilon;
};
//==============================================================================
} // namespace sarathi
//==============================================================================

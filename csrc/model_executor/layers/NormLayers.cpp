#include "NormLayers.h"
#include "ops.h"
//==============================================================================
using namespace sarathi;
//==============================================================================
RMSNorm::RMSNorm(
    const torch::Tensor& weight,
    double fVarianceEpsilon
) : 
    m_weight(weight),
    m_fVarianceEpsilon(fVarianceEpsilon)
{}
//==============================================================================
torch::Tensor RMSNorm::Forward(const torch::Tensor& input /*[in]*/) const
{
    torch::Tensor out = torch::empty_like(input);
    rms_norm(out, input, m_weight, m_fVarianceEpsilon);
    return out;
}
//==============================================================================

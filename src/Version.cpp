#include "minitensor/Version.h"

namespace minitensor {

const char* version() noexcept
{
    return MINITENSOR_VERSION_STRING;
}

bool openmp_enabled() noexcept
{
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

}

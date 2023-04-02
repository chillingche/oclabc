#ifndef _TYPE_H_
#define _TYPE_H_

namespace abc {

typedef enum UT_RANDOM_TYPE {
    UT_INIT_RANDOM,  // random
    UT_INIT_NEG,     // random & < 0
    UT_INIT_POS,     // random & > 0
    UT_INIT_ZERO     // 0
} UT_RANDOM_TYPE;

struct dims4d {
    int n, c, h, w;
};

}  // namespace abc
#endif

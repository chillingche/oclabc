#include <stdio.h>

#ifndef _LOG_H_
#define _LOG_H_

#ifndef TAG
#define TAG "OCLABC"
#endif

#define LOGD(MSG, ...) do { \
    printf("[DEBUG] [OCLABC][" TAG "] " MSG "\n", ##__VA_ARGS__); \
} while (0);

#define LOGI(MSG, ...) do { \
    printf("[INFO] [OCLABC][" TAG "] " MSG "\n", ##__VA_ARGS__); \
} while (0);

#define LOGW(MSG, ...) do { \
    printf("[WARN] [OCLABC][" TAG "] " MSG "\n", ##__VA_ARGS__); \
} while (0);

#define LOGE(MSG, ...) do { \
    printf("[ERROR] [OCLABC][" TAG "] " MSG "\n", ##__VA_ARGS__); \
} while (0);

#define CHECK_ERROR_NO_RETURN(COND, MSG, ...) do { \
    (COND) ? (NULL) : printf("[ERROR] [OCLABC][" TAG "] " MSG "\n", ##__VA_ARGS__); \
} while (0);


#endif
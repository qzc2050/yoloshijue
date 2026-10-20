#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32h7xx_hal.h"

#define LED1_Pin        GPIO_PIN_2
#define LED1_GPIO_Port  GPIOE

#define USART1_BAUDRATE 115200U

void Error_Handler(void);

#ifdef __cplusplus
}
#endif

#endif

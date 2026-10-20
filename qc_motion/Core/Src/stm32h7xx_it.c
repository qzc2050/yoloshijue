#include "main.h"
#include "stm32h7xx_it.h"
#include "usart.h"
#include "cmd.h"

void NMI_Handler(void)
{
    while (1) {
    }
}

void HardFault_Handler(void)
{
    while (1) {
    }
}

void MemManage_Handler(void)
{
    while (1) {
    }
}

void BusFault_Handler(void)
{
    while (1) {
    }
}

void UsageFault_Handler(void)
{
    while (1) {
    }
}

void SVC_Handler(void)
{
}

void DebugMon_Handler(void)
{
}

void PendSV_Handler(void)
{
}

void SysTick_Handler(void)
{
    HAL_IncTick();
}

void USART1_IRQHandler(void)
{
    if (__HAL_UART_GET_FLAG(&huart1, UART_FLAG_RXNE) != RESET) {
        uint8_t ch = (uint8_t)(USART1->RDR & 0xFF);
        Cmd_RxByte(ch);
        while ((USART1->ISR & USART_ISR_TXE_TXFNF) == 0) {
        }
        USART1->TDR = ch;
    }
    if (__HAL_UART_GET_FLAG(&huart1, UART_FLAG_ORE) != RESET) {
        __HAL_UART_CLEAR_OREFLAG(&huart1);
        (void)USART1->RDR;
    }
}

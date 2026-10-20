#include "cmd.h"
#include "motion.h"
#include <stdio.h>
#include <string.h>

#define RX_SIZE   64U
#define LINE_SIZE 32U

static volatile uint8_t rx_buf[RX_SIZE];
static volatile uint16_t rx_w;
static volatile uint16_t rx_r;
static volatile uint8_t stop_req;

void Cmd_RxByte(uint8_t ch)
{
    uint16_t next;

    if (ch == 's' || ch == 'S') {
        stop_req = 1U;
        Motion_RequestStop();
    }
    next = (uint16_t)((rx_w + 1U) % RX_SIZE);
    if (next != rx_r) {
        rx_buf[rx_w] = ch;
        rx_w = next;
    }
}

static int rx_get(uint8_t *out)
{
    if (rx_r == rx_w) {
        return 0;
    }
    *out = rx_buf[rx_r];
    rx_r = (uint16_t)((rx_r + 1U) % RX_SIZE);
    return 1;
}

static void trim_lower(char *s)
{
    char *d = s;
    char *src = s;

    while (*src == ' ' || *src == '\t') {
        src++;
    }
    while (*src) {
        char c = *src++;
        if (c >= 'A' && c <= 'Z') {
            c = (char)(c + 32);
        }
        *d++ = c;
    }
    while (d > s && (d[-1] == ' ' || d[-1] == '\t')) {
        d--;
    }
    *d = '\0';
}

static const char *skip_sp(const char *s)
{
    while (*s == ' ' || *s == '\t') {
        s++;
    }
    return s;
}

/* Millimeters with up to 3 decimal places, returned as mm * 1000. */
static int parse_mm(const char **ps, int32_t *milli)
{
    const char *s = skip_sp(*ps);
    int neg = 0;
    int any = 0;
    int fdigits = 0;
    int32_t ip = 0;
    int32_t frac = 0;

    if (*s == '+') {
        s++;
    } else if (*s == '-') {
        neg = 1;
        s++;
    }
    while (*s >= '0' && *s <= '9') {
        any = 1;
        if (ip > 100000) {
            return 0;
        }
        ip = ip * 10 + (*s - '0');
        s++;
    }
    if (*s == '.') {
        s++;
        while (*s >= '0' && *s <= '9') {
            any = 1;
            if (fdigits < 3) {
                frac = frac * 10 + (*s - '0');
                fdigits++;
            }
            s++;
        }
    }
    if (!any) {
        return 0;
    }
    while (fdigits < 3) {
        frac *= 10;
        fdigits++;
    }
    *milli = ip * 1000 + frac;
    if (neg) {
        *milli = -*milli;
    }
    *ps = s;
    return 1;
}

void Cmd_PrintHelp(void)
{
    printf(
        "x+ x-   socket C / J8, 2mm\r\n"
        "y+ y-   socket B / J7, 2mm\r\n"
        "z+ z-   socket A / J6, 2mm\r\n"
        "zero    here is x=0 y=0\r\n"
        "pos     print mm from zero\r\n"
        "m x y   move to mm, x then y\r\n"
        "s       stop now\r\n");
}

static void handle_line(char *line)
{
    trim_lower(line);
    if (line[0] == '\0') {
        return;
    }

    if (strcmp(line, "x+") == 0) {
        Motion_Jog(AXIS_X, +1);
    } else if (strcmp(line, "x-") == 0) {
        Motion_Jog(AXIS_X, -1);
    } else if (strcmp(line, "y+") == 0) {
        Motion_Jog(AXIS_Y, +1);
    } else if (strcmp(line, "y-") == 0) {
        Motion_Jog(AXIS_Y, -1);
    } else if (strcmp(line, "z+") == 0) {
        Motion_Jog(AXIS_Z, +1);
    } else if (strcmp(line, "z-") == 0) {
        Motion_Jog(AXIS_Z, -1);
    } else if (strcmp(line, "s") == 0 || strcmp(line, "stop") == 0) {
        /* already handled by stop_req */
    } else if (strcmp(line, "zero") == 0) {
        Motion_Zero();
    } else if (strcmp(line, "pos") == 0) {
        Motion_PrintPos();
    } else if (line[0] == 'm' && (line[1] == ' ' || line[1] == '\t')) {
        const char *p = line + 1;
        int32_t x_mm;
        int32_t y_mm;

        if (!parse_mm(&p, &x_mm) || !parse_mm(&p, &y_mm) || *skip_sp(p) != '\0') {
            printf("use  m <x> <y>\r\n");
        } else {
            Motion_MoveAbs(x_mm, y_mm);
        }
    } else if (strcmp(line, "h") == 0 || strcmp(line, "help") == 0 || strcmp(line, "?") == 0) {
        Cmd_PrintHelp();
    } else {
        printf("unk  %s\r\n", line);
    }
}

void Cmd_Poll(void)
{
    static char line[LINE_SIZE];
    static uint8_t len;
    uint8_t ch;

    if (stop_req) {
        stop_req = 0U;
        Motion_Stop();
        printf("stop\r\n");
    }

    while (rx_get(&ch)) {
        if (ch == '\r' || ch == '\n') {
            if (len > 0U) {
                line[len] = '\0';
                handle_line(line);
                len = 0U;
            }
            continue;
        }
        if (len < (LINE_SIZE - 1U)) {
            line[len++] = (char)ch;
        } else {
            len = 0U;
        }
    }
}

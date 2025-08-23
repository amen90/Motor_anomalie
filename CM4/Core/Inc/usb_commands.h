#ifndef __USB_COMMANDS_H
#define __USB_COMMANDS_H

#include "main.h"
#include <stdint.h>
#include <stdbool.h>

/* USB Command buffer size */
#define USB_CMD_BUFFER_SIZE     64
#define USB_RESPONSE_BUFFER_SIZE 128

/* USB Command types */
typedef enum {
    CMD_UNKNOWN = 0,
    CMD_START_NORMAL,
    CMD_START_IMBALANCE,
    CMD_START_BEARING,
    CMD_START_MISALIGN,
    CMD_STOP_COLLECTION,
    CMD_GET_DATA,
    CMD_GET_STATUS,
    CMD_RESET_SYSTEM
} usb_command_type_t;

/* USB Command structure */
typedef struct {
    usb_command_type_t type;
    char raw_command[USB_CMD_BUFFER_SIZE];
    bool is_valid;
} usb_command_t;

/* Function prototypes */
void usb_commands_init(void);
bool usb_parse_command(const char* input, usb_command_t* cmd);
void usb_execute_command(const usb_command_t* cmd);
void usb_send_response(const char* response);
void usb_process_input_buffer(void);

/* USB CDC callback functions */
void usb_cdc_receive_callback(uint8_t* buffer, uint32_t length);

#endif /* __USB_COMMANDS_H */

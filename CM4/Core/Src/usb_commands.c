#include "usb_commands.h"
#include "ai_data_collection.h"
#include <string.h>
#include <stdio.h>

/* Static variables for USB command processing */
static char usb_input_buffer[USB_CMD_BUFFER_SIZE];
static volatile uint32_t buffer_index = 0;
static volatile bool command_ready = false;

/* Initialize USB command system */
void usb_commands_init(void)
{
    memset(usb_input_buffer, 0, sizeof(usb_input_buffer));
    buffer_index = 0;
    command_ready = false;
}

/* Parse incoming command string */
bool usb_parse_command(const char* input, usb_command_t* cmd)
{
    if (!input || !cmd) {
        return false;
    }
    
    memset(cmd, 0, sizeof(usb_command_t));
    strncpy(cmd->raw_command, input, USB_CMD_BUFFER_SIZE - 1);
    
    /* Parse command type */
    if (strncmp(input, "START_NORMAL", 12) == 0) {
        cmd->type = CMD_START_NORMAL;
        cmd->is_valid = true;
    } else if (strncmp(input, "START_IMBALANCE", 15) == 0) {
        cmd->type = CMD_START_IMBALANCE;
        cmd->is_valid = true;
    } else if (strncmp(input, "START_BEARING", 13) == 0) {
        cmd->type = CMD_START_BEARING;
        cmd->is_valid = true;
    } else if (strncmp(input, "START_MISALIGN", 14) == 0) {
        cmd->type = CMD_START_MISALIGN;
        cmd->is_valid = true;
    } else if (strncmp(input, "STOP", 4) == 0) {
        cmd->type = CMD_STOP_COLLECTION;
        cmd->is_valid = true;
    } else if (strncmp(input, "GET_DATA", 8) == 0) {
        cmd->type = CMD_GET_DATA;
        cmd->is_valid = true;
    } else if (strncmp(input, "STATUS", 6) == 0) {
        cmd->type = CMD_GET_STATUS;
        cmd->is_valid = true;
    } else if (strncmp(input, "RESET", 5) == 0) {
        cmd->type = CMD_RESET_SYSTEM;
        cmd->is_valid = true;
    } else {
        cmd->type = CMD_UNKNOWN;
        cmd->is_valid = false;
    }
    
    return cmd->is_valid;
}

/* Execute parsed command */
void usb_execute_command(const usb_command_t* cmd)
{
    if (!cmd || !cmd->is_valid) {
        usb_send_response("ERROR: Invalid command");
        return;
    }
    
    switch (cmd->type) {
        case CMD_START_NORMAL:
            if (ai_start_collection(MOTOR_NORMAL)) {
                usb_send_response("OK: Started normal motor data collection");
            } else {
                usb_send_response("ERROR: Failed to start collection");
            }
            break;
            
        case CMD_START_IMBALANCE:
            if (ai_start_collection(MOTOR_IMBALANCE)) {
                usb_send_response("OK: Started imbalance motor data collection");
            } else {
                usb_send_response("ERROR: Failed to start collection");
            }
            break;
            
        case CMD_START_BEARING:
            if (ai_start_collection(MOTOR_BEARING_FAULT)) {
                usb_send_response("OK: Started bearing fault data collection");
            } else {
                usb_send_response("ERROR: Failed to start collection");
            }
            break;
            
        case CMD_START_MISALIGN:
            if (ai_start_collection(MOTOR_MISALIGNMENT)) {
                usb_send_response("OK: Started misalignment data collection");
            } else {
                usb_send_response("ERROR: Failed to start collection");
            }
            break;
            
        case CMD_STOP_COLLECTION:
            if (ai_stop_collection()) {
                usb_send_response("OK: Collection stopped");
            } else {
                usb_send_response("ERROR: Failed to stop collection");
            }
            break;
            
        case CMD_GET_DATA:
            {
                ai_training_sample_t sample;
                if (ai_get_sample_data(&sample)) {
                    ai_send_sample_via_usb(&sample);
                } else {
                    usb_send_response("ERROR: No data available");
                }
            }
            break;
            
        case CMD_GET_STATUS:
            {
                ai_collection_status_t status = ai_get_collection_status();
                char response[64];
                snprintf(response, sizeof(response), "STATUS: %d", (int)status);
                usb_send_response(response);
            }
            break;
            
        case CMD_RESET_SYSTEM:
            ai_reset_collection();
            usb_send_response("OK: System reset");
            break;
            
        default:
            usb_send_response("ERROR: Unknown command");
            break;
    }
}

/* Send response via USB */
void usb_send_response(const char* response)
{
    if (response) {
        printf("%s\r\n", response);
    }
}

/* Process input buffer for complete commands */
void usb_process_input_buffer(void)
{
    if (command_ready) {
        usb_command_t cmd;
        
        /* Null-terminate the command */
        usb_input_buffer[buffer_index] = '\0';
        
        /* Parse and execute command */
        if (usb_parse_command(usb_input_buffer, &cmd)) {
            usb_execute_command(&cmd);
        } else {
            usb_send_response("ERROR: Invalid command format");
        }
        
        /* Reset buffer */
        memset(usb_input_buffer, 0, sizeof(usb_input_buffer));
        buffer_index = 0;
        command_ready = false;
    }
}

/* USB CDC receive callback */
void usb_cdc_receive_callback(uint8_t* buffer, uint32_t length)
{
    for (uint32_t i = 0; i < length; i++) {
        char c = (char)buffer[i];
        
        /* Check for command terminator */
        if (c == '\r' || c == '\n') {
            if (buffer_index > 0) {
                command_ready = true;
            }
        } else if (c >= ' ' && c <= '~') {  /* Printable ASCII */
            if (buffer_index < (USB_CMD_BUFFER_SIZE - 1)) {
                usb_input_buffer[buffer_index++] = c;
            }
        }
    }
}

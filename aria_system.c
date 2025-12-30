#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#define VERSION "1.0"
#define AI_NAME "ARIA"
#define AI_FULL_NAME "Adaptive Recursive Intelligence Assistant"
#define GITHUB_USER "sugarypumpkin822"
#define GITHUB_REPO "ai-chat-system"
#define MODEL_PATH "R:\\aria_model_v1.pt"
#define DATASET_PATH "R:\\Datasets\\"
#define CODE_PATH "R:\\Datasets\\Code\\"
#define CHAT_HISTORY_PATH "R:\\Datasets\\chat_history.txt"
#define MAX_LAYERS 12
#define MAX_NEURONS 2048
#define BUFFER_SIZE 131072
#define MAX_TOKENS 50000
#define MAX_CHAT_HISTORY 100

// Color scheme (ChatGPT-like)
#define COLOR_BG RGB(52, 53, 65)
#define COLOR_SIDEBAR RGB(32, 33, 35)
#define COLOR_INPUT_BG RGB(64, 65, 79)
#define COLOR_USER_MSG RGB(86, 88, 105)
#define COLOR_AI_MSG RGB(68, 70, 84)
#define COLOR_TEXT RGB(236, 236, 241)
#define COLOR_ACCENT RGB(16, 163, 127)
#define COLOR_BUTTON RGB(25, 195, 125)
#define COLOR_BUTTON_HOVER RGB(26, 127, 100)

// Enhanced Neural Network Structures
typedef struct {
    float* weights;
    float* biases;
    float* activations;
    float* deltas;
    float* weight_momentum;
    float* bias_momentum;
    int input_size;
    int output_size;
    float dropout_rate;
} Layer;

typedef struct {
    Layer layers[MAX_LAYERS];
    int num_layers;
    float learning_rate;
    float momentum;
    float weight_decay;
    int vocab_size;
    char** vocabulary;
    int* token_frequencies;
    float* token_embeddings;
    int embedding_dim;
    long long training_steps;
    float temperature;
} NeuralNetwork;

typedef struct {
    char** tokens;
    int* token_ids;
    int length;
    float* embeddings;
} TokenizedText;

typedef struct {
    char* message;
    char* response;
    time_t timestamp;
} ChatEntry;

typedef struct {
    ChatEntry entries[MAX_CHAT_HISTORY];
    int count;
} ChatHistory;

// GUI Structures
typedef struct {
    HWND hwnd;
    HWND chat_display;
    HWND input_box;
    HWND send_button;
    HWND train_button;
    HWND scrape_button;
    HWND update_button;
    HWND load_button;
    HWND save_button;
    HWND clear_button;
    HWND status_bar;
    HWND sidebar;
    HWND model_info;
    NeuralNetwork* nn;
    ChatHistory history;
    HFONT chat_font;
    HFONT button_font;
    HFONT status_font;
    HBRUSH bg_brush;
    HBRUSH sidebar_brush;
    HBRUSH input_brush;
    HBRUSH button_brush;
    int button_hover;
} AppWindow;

// Global Variables
AppWindow g_app;
char g_status_text[512] = "ARIA initialized - Ready to assist!";

// ============ ENHANCED MATH FUNCTIONS ============

float leaky_relu(float x, float alpha) {
    return x > 0 ? x : alpha * x;
}

float leaky_relu_derivative(float x, float alpha) {
    return x > 0 ? 1.0f : alpha;
}

float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

float gelu_derivative(float x) {
    float tanh_arg = 0.7978845608f * (x + 0.044715f * x * x * x);
    float sech2 = 1.0f - tanhf(tanh_arg) * tanhf(tanh_arg);
    return 0.5f * (1.0f + tanhf(tanh_arg)) + 
           0.5f * x * sech2 * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x) {
    return x * (1.0f - x);
}

float softmax(float* input, int size, int index) {
    float max = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max) max = input[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += expf(input[i] - max);
    }
    
    return expf(input[index] - max) / sum;
}

// ============ CUSTOM MESSAGE BOX ============

void CustomMessageBox(HWND hwnd, const char* message, const char* title) {
    HWND msgBox = CreateWindowEx(
        WS_EX_TOPMOST | WS_EX_DLGMODALFRAME,
        "STATIC",
        title,
        WS_POPUP | WS_VISIBLE | WS_BORDER | SS_CENTER,
        (GetSystemMetrics(SM_CXSCREEN) - 400) / 2,
        (GetSystemMetrics(SM_CYSCREEN) - 200) / 2,
        400, 200,
        hwnd, NULL, GetModuleHandle(NULL), NULL
    );
    
    CreateWindowEx(
        0, "STATIC", message,
        WS_CHILD | WS_VISIBLE | SS_CENTER,
        20, 40, 360, 80,
        msgBox, NULL, GetModuleHandle(NULL), NULL
    );
    
    CreateWindowEx(
        0, "BUTTON", "OK",
        WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
        160, 140, 80, 35,
        msgBox, (HMENU)IDOK, GetModuleHandle(NULL), NULL
    );
    
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        if (msg.message == WM_COMMAND && LOWORD(msg.wParam) == IDOK) {
            DestroyWindow(msgBox);
            break;
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

// ============ ENHANCED NETWORK IMPLEMENTATION ============

int InitializeNetwork() {
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
}

int SendHTTPRequest(const char* host, const char* path, char* response, int max_len) {
    SOCKET sock;
    struct sockaddr_in server;
    struct hostent* he;
    char request[4096];
    int received = 0;
    
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET) return 0;
    
    DWORD timeout = 15000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
    
    he = gethostbyname(host);
    if (!he) {
        closesocket(sock);
        return 0;
    }
    
    server.sin_family = AF_INET;
    server.sin_port = htons(80);
    server.sin_addr = *((struct in_addr*)he->h_addr);
    
    if (connect(sock, (struct sockaddr*)&server, sizeof(server)) < 0) {
        closesocket(sock);
        return 0;
    }
    
    sprintf(request, 
        "GET %s HTTP/1.1\r\n"
        "Host: %s\r\n"
        "User-Agent: %s/%s\r\n"
        "Accept: text/html,application/json,*/*\r\n"
        "Accept-Language: en-US,en;q=0.9\r\n"
        "Connection: close\r\n\r\n",
        path, host, AI_NAME, VERSION
    );
    
    send(sock, request, (int)strlen(request), 0);
    
    response[0] = '\0';
    while (received < max_len - 1) {
        int bytes = recv(sock, response + received, max_len - received - 1, 0);
        if (bytes <= 0) break;
        received += bytes;
    }
    response[received] = '\0';
    
    closesocket(sock);
    return received;
}

// ============ ENHANCED WEB SCRAPING ENGINE ============

typedef struct {
    char** urls;
    int count;
    int capacity;
} URLList;

URLList* CreateURLList() {
    URLList* list = (URLList*)malloc(sizeof(URLList));
    list->capacity = 500;
    list->count = 0;
    list->urls = (char**)malloc(sizeof(char*) * list->capacity);
    return list;
}

void AddURL(URLList* list, const char* url) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->urls = (char**)realloc(list->urls, sizeof(char*) * list->capacity);
    }
    list->urls[list->count] = _strdup(url);
    list->count++;
}

void FreeURLList(URLList* list) {
    for (int i = 0; i < list->count; i++) {
        free(list->urls[i]);
    }
    free(list->urls);
    free(list);
}

char* ExtractTextFromHTML(const char* html) {
    int len = (int)strlen(html);
    char* text = (char*)malloc(len + 1);
    int pos = 0;
    int in_tag = 0;
    int in_script = 0;
    int in_style = 0;
    
    for (int i = 0; i < len; i++) {
        if (html[i] == '<') {
            in_tag = 1;
            if (i + 7 < len && strncmp(&html[i], "<script", 7) == 0) in_script = 1;
            if (i + 8 < len && strncmp(&html[i], "</script", 8) == 0) in_script = 0;
            if (i + 6 < len && strncmp(&html[i], "<style", 6) == 0) in_style = 1;
            if (i + 7 < len && strncmp(&html[i], "</style", 7) == 0) in_style = 0;
        } else if (html[i] == '>') {
            in_tag = 0;
        } else if (!in_tag && !in_script && !in_style) {
            if (html[i] != '\n' && html[i] != '\r' && html[i] != '\t') {
                text[pos++] = html[i];
            } else if (pos > 0 && text[pos - 1] != ' ') {
                text[pos++] = ' ';
            }
        }
    }
    text[pos] = '\0';
    return text;
}

URLList* ExtractURLsFromHTML(const char* html, const char* base_url) {
    URLList* urls = CreateURLList();
    const char* ptr = html;
    (void)base_url;
    
    while ((ptr = strstr(ptr, "href=\"")) != NULL) {
        ptr += 6;
        const char* end = strchr(ptr, '"');
        if (end) {
            int url_len = (int)(end - ptr);
            if (url_len > 0 && url_len < 1024) {
                char url[1024];
                strncpy(url, ptr, url_len);
                url[url_len] = '\0';
                
                if (strncmp(url, "http", 4) == 0 && !strstr(url, "javascript:")) {
                    AddURL(urls, url);
                }
            }
        }
        ptr = end ? end + 1 : ptr + 1;
    }
    
    return urls;
}

const char* DetermineFileType(const char* content) {
    if (strstr(content, "#include") || strstr(content, "int main") || strstr(content, "void ")) return "C";
    if (strstr(content, "def ") || strstr(content, "import ") || strstr(content, "class ")) return "Python";
    if (strstr(content, "function") || strstr(content, "const ") || strstr(content, "let ")) return "JavaScript";
    if (strstr(content, "public class") || strstr(content, "public static void")) return "Java";
    if (strstr(content, "<!DOCTYPE") || strstr(content, "<html")) return "HTML";
    if (strstr(content, "SELECT") || strstr(content, "INSERT INTO")) return "SQL";
    if (strstr(content, "#include <iostream>") || strstr(content, "std::")) return "CPP";
    if (strstr(content, "using System") || strstr(content, "namespace ")) return "CSharp";
    if (strstr(content, "package ") || strstr(content, "func ")) return "Go";
    if (strstr(content, "fn ") || strstr(content, "let mut")) return "Rust";
    return "Text";
}

void SaveScrapedData(const char* content, const char* url) {
    const char* file_type = DetermineFileType(content);
    char filename[1024];
    time_t t = time(NULL);
    
    // Determine if it's code or general content
    int is_code = (strcmp(file_type, "C") == 0 || 
                   strcmp(file_type, "Python") == 0 || 
                   strcmp(file_type, "JavaScript") == 0 ||
                   strcmp(file_type, "Java") == 0 || 
                   strcmp(file_type, "CPP") == 0 || 
                   strcmp(file_type, "CSharp") == 0 ||
                   strcmp(file_type, "Go") == 0 || 
                   strcmp(file_type, "Rust") == 0 || 
                   strcmp(file_type, "HTML") == 0 ||
                   strcmp(file_type, "SQL") == 0);
    
    // Save code to R:\Datasets\Code, other content to R:\Datasets
    if (is_code) {
        sprintf(filename, "%s%s_%ld.txt", CODE_PATH, file_type, (long)t);
    } else {
        sprintf(filename, "%s%s_%ld.txt", DATASET_PATH, file_type, (long)t);
    }
    
    FILE* f = fopen(filename, "w");
    if (f) {
        fprintf(f, "=== %s Data Scraped by %s ===\n", AI_NAME, AI_FULL_NAME);
        fprintf(f, "URL: %s\n", url);
        fprintf(f, "Type: %s\n", file_type);
        fprintf(f, "Location: %s\n", is_code ? "Code Repository" : "General Data");
        fprintf(f, "Timestamp: %ld\n", (long)t);
        fprintf(f, "================================\n\n");
        fprintf(f, "%s", content);
        fclose(f);
    }
}

void ScrapeWeb(const char* start_url, int max_pages) {
    URLList* to_visit = CreateURLList();
    URLList* visited = CreateURLList();
    AddURL(to_visit, start_url);
    
    int pages_scraped = 0;
    char* response = (char*)malloc(BUFFER_SIZE);
    
    while (to_visit->count > 0 && pages_scraped < max_pages) {
        char* current_url = to_visit->urls[--to_visit->count];
        
        int already_visited = 0;
        for (int i = 0; i < visited->count; i++) {
            if (strcmp(visited->urls[i], current_url) == 0) {
                already_visited = 1;
                break;
            }
        }
        
        if (already_visited) {
            free(current_url);
            continue;
        }
        AddURL(visited, current_url);
        
        char host[512], path[1024];
        if (sscanf(current_url, "http://%511[^/]%1023s", host, path) >= 1) {
            if (path[0] == '\0') strcpy(path, "/");
            
            sprintf(g_status_text, "%s is scraping: %s (%d/%d)", AI_NAME, host, pages_scraped + 1, max_pages);
            SetWindowText(g_app.status_bar, g_status_text);
            
            if (SendHTTPRequest(host, path, response, BUFFER_SIZE)) {
                char* body = strstr(response, "\r\n\r\n");
                if (body) {
                    body += 4;
                    char* text = ExtractTextFromHTML(body);
                    SaveScrapedData(text, current_url);
                    free(text);
                    
                    URLList* new_urls = ExtractURLsFromHTML(body, current_url);
                    for (int i = 0; i < new_urls->count && to_visit->count < 200; i++) {
                        AddURL(to_visit, new_urls->urls[i]);
                    }
                    FreeURLList(new_urls);
                    
                    pages_scraped++;
                }
            }
        }
        
        free(current_url);
        Sleep(100);
    }
    
    free(response);
    FreeURLList(to_visit);
    FreeURLList(visited);
}

// ============ ENHANCED NEURAL NETWORK ============

Layer* CreateLayer(int input_size, int output_size, float dropout) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->dropout_rate = dropout;
    
    layer->weights = (float*)malloc(input_size * output_size * sizeof(float));
    layer->biases = (float*)malloc(output_size * sizeof(float));
    layer->activations = (float*)malloc(output_size * sizeof(float));
    layer->deltas = (float*)malloc(output_size * sizeof(float));
    layer->weight_momentum = (float*)calloc(input_size * output_size, sizeof(float));
    layer->bias_momentum = (float*)calloc(output_size, sizeof(float));
    
    float scale = sqrtf(2.0f / input_size);
    for (int i = 0; i < input_size * output_size; i++) {
        layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    for (int i = 0; i < output_size; i++) {
        layer->biases[i] = 0.01f;
    }
    
    return layer;
}

NeuralNetwork* CreateNeuralNetwork(int* layer_sizes, int num_layers) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers - 1;
    nn->learning_rate = 0.0005f;
    nn->momentum = 0.9f;
    nn->weight_decay = 0.0001f;
    nn->vocab_size = 0;
    nn->embedding_dim = 128;
    nn->temperature = 0.8f;
    nn->training_steps = 0;
    nn->vocabulary = (char**)malloc(MAX_TOKENS * sizeof(char*));
    nn->token_frequencies = (int*)calloc(MAX_TOKENS, sizeof(int));
    nn->token_embeddings = (float*)malloc(MAX_TOKENS * nn->embedding_dim * sizeof(float));
    
    for (int i = 0; i < MAX_TOKENS * nn->embedding_dim; i++) {
        nn->token_embeddings[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    for (int i = 0; i < num_layers - 1; i++) {
        float dropout = (i < num_layers - 2) ? 0.1f : 0.0f;
        nn->layers[i] = *CreateLayer(layer_sizes[i], layer_sizes[i + 1], dropout);
    }
    
    return nn;
}

TokenizedText* Tokenize(const char* text, NeuralNetwork* nn) {
    TokenizedText* tokens = (TokenizedText*)malloc(sizeof(TokenizedText));
    tokens->tokens = (char**)malloc(MAX_TOKENS * sizeof(char*));
    tokens->token_ids = (int*)malloc(MAX_TOKENS * sizeof(int));
    tokens->embeddings = (float*)malloc(MAX_TOKENS * nn->embedding_dim * sizeof(float));
    tokens->length = 0;
    
    char* text_copy = _strdup(text);
    
    for (char* p = text_copy; *p; p++) {
        *p = (char)tolower((unsigned char)*p);
    }
    
    char* token = strtok(text_copy, " \t\n\r.,;:!?()[]{}\"'<>/\\|");
    
    while (token && tokens->length < MAX_TOKENS - 1) {
        if (strlen(token) > 0) {
            int token_id = -1;
            for (int i = 0; i < nn->vocab_size; i++) {
                if (strcmp(nn->vocabulary[i], token) == 0) {
                    token_id = i;
                    break;
                }
            }
            
            if (token_id == -1 && nn->vocab_size < MAX_TOKENS - 1) {
                token_id = nn->vocab_size;
                nn->vocabulary[nn->vocab_size] = _strdup(token);
                nn->vocab_size++;
            }
            
            if (token_id != -1) {
                tokens->tokens[tokens->length] = _strdup(token);
                tokens->token_ids[tokens->length] = token_id;
                nn->token_frequencies[token_id]++;
                
                for (int i = 0; i < nn->embedding_dim; i++) {
                    tokens->embeddings[tokens->length * nn->embedding_dim + i] = 
                        nn->token_embeddings[token_id * nn->embedding_dim + i];
                }
                
                tokens->length++;
            }
        }
        
        token = strtok(NULL, " \t\n\r.,;:!?()[]{}\"'<>/\\|");
    }
    
    free(text_copy);
    return tokens;
}

void Forward(NeuralNetwork* nn, float* input, int training) {
    float* current_input = input;
    
    for (int l = 0; l < nn->num_layers; l++) {
        Layer* layer = &nn->layers[l];
        
        for (int j = 0; j < layer->output_size; j++) {
            float sum = layer->biases[j];
            for (int i = 0; i < layer->input_size; i++) {
                sum += current_input[i] * layer->weights[i * layer->output_size + j];
            }
            
            if (l == nn->num_layers - 1) {
                layer->activations[j] = sum;
            } else {
                layer->activations[j] = gelu(sum);
                
                if (training && layer->dropout_rate > 0) {
                    if ((float)rand() / RAND_MAX < layer->dropout_rate) {
                        layer->activations[j] = 0;
                    } else {
                        layer->activations[j] /= (1.0f - layer->dropout_rate);
                    }
                }
            }
        }
        
        current_input = layer->activations;
    }
}

void Backward(NeuralNetwork* nn, float* input, float* target) {
    Layer* output_layer = &nn->layers[nn->num_layers - 1];
    
    for (int i = 0; i < output_layer->output_size; i++) {
        float prob = softmax(output_layer->activations, output_layer->output_size, i);
        output_layer->deltas[i] = prob - target[i];
    }
    
    for (int l = nn->num_layers - 2; l >= 0; l--) {
        Layer* layer = &nn->layers[l];
        Layer* next_layer = &nn->layers[l + 1];
        
        for (int i = 0; i < layer->output_size; i++) {
            float error = 0.0f;
            for (int j = 0; j < next_layer->output_size; j++) {
                error += next_layer->deltas[j] * next_layer->weights[i * next_layer->output_size + j];
            }
            layer->deltas[i] = error * gelu_derivative(layer->activations[i]);
        }
    }
    
    float* current_input = input;
    for (int l = 0; l < nn->num_layers; l++) {
        Layer* layer = &nn->layers[l];
        
        for (int i = 0; i < layer->input_size; i++) {
            for (int j = 0; j < layer->output_size; j++) {
                int idx = i * layer->output_size + j;
                float gradient = layer->deltas[j] * current_input[i];
                
                gradient += nn->weight_decay * layer->weights[idx];
                
                layer->weight_momentum[idx] = nn->momentum * layer->weight_momentum[idx] - 
                                              nn->learning_rate * gradient;
                layer->weights[idx] += layer->weight_momentum[idx];
            }
        }
        
        for (int j = 0; j < layer->output_size; j++) {
            layer->bias_momentum[j] = nn->momentum * layer->bias_momentum[j] - 
                                      nn->learning_rate * layer->deltas[j];
            layer->biases[j] += layer->bias_momentum[j];
        }
        
        current_input = layer->activations;
    }
    
    nn->training_steps++;
}

void TrainOnText(NeuralNetwork* nn, const char* text, int epochs) {
    TokenizedText* tokens = Tokenize(text, nn);
    
    if (tokens->length < 2) {
        free(tokens);
        return;
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int samples = 0;
        
        for (int i = 0; i < tokens->length - 1; i++) {
            float* input = (float*)calloc(nn->vocab_size, sizeof(float));
            float* target = (float*)calloc(nn->vocab_size, sizeof(float));
            
            input[tokens->token_ids[i]] = 1.0f;
            target[tokens->token_ids[i + 1]] = 1.0f;
            
            Forward(nn, input, 1);
            Backward(nn, input, target);
            
            Layer* output = &nn->layers[nn->num_layers - 1];
            for (int j = 0; j < output->output_size; j++) {
                float diff = target[j] - softmax(output->activations, output->output_size, j);
                total_loss += diff * diff;
            }
            
            samples++;
            free(input);
            free(target);
        }
        
        if (epoch % 5 == 0) {
            float avg_loss = total_loss / (samples > 0 ? samples : 1);
            sprintf(g_status_text, "%s training - Epoch %d/%d | Loss: %.4f | Vocab: %d tokens", 
                    AI_NAME, epoch + 1, epochs, avg_loss, nn->vocab_size);
            SetWindowText(g_app.status_bar, g_status_text);
        }
    }
    
    free(tokens->tokens);
    free(tokens->token_ids);
    free(tokens->embeddings);
    free(tokens);
}

char* GenerateText(NeuralNetwork* nn, const char* seed, int length) {
    char* result = (char*)malloc(length * 100);
    result[0] = '\0';
    
    TokenizedText* seed_tokens = Tokenize(seed, nn);
    if (seed_tokens->length == 0) {
        strcpy(result, "I need more context to generate a response!");
        free(seed_tokens->tokens);
        free(seed_tokens->token_ids);
        free(seed_tokens->embeddings);
        free(seed_tokens);
        return result;
    }
    
    int current_token = seed_tokens->token_ids[seed_tokens->length - 1];
    strcat(result, seed);
    strcat(result, " ");
    
    for (int i = 0; i < length; i++) {
        float* input = (float*)calloc(nn->vocab_size, sizeof(float));
        input[current_token] = 1.0f;
        
        Forward(nn, input, 0);
        
        Layer* output = &nn->layers[nn->num_layers - 1];
        
        float probs[MAX_NEURONS];
        float sum = 0.0f;
        for (int j = 0; j < nn->vocab_size && j < MAX_NEURONS; j++) {
            probs[j] = expf(output->activations[j] / nn->temperature);
            sum += probs[j];
        }
        
        float r = (float)rand() / RAND_MAX * sum;
        float cumsum = 0.0f;
        int next_token = 0;
        
        for (int j = 0; j < nn->vocab_size && j < MAX_NEURONS; j++) {
            cumsum += probs[j];
            if (cumsum >= r) {
                next_token = j;
                break;
            }
        }
        
        if (next_token < nn->vocab_size && nn->vocabulary[next_token]) {
            strcat(result, nn->vocabulary[next_token]);
            strcat(result, " ");
        }
        
        current_token = next_token;
        free(input);
    }
    
    free(seed_tokens->tokens);
    free(seed_tokens->token_ids);
    free(seed_tokens->embeddings);
    free(seed_tokens);
    
    return result;
}

// ============ ARIA PERSONALITY ============

const char* GetARIAGreeting() {
    const char* greetings[] = {
        "Hello! I'm ARIA, your Adaptive Recursive Intelligence Assistant. How can I help you today?",
        "Hi there! ARIA here, ready to learn and assist you!",
        "Greetings! I'm ARIA, and I'm excited to chat with you!",
        "Hey! ARIA at your service. What would you like to explore today?"
    };
    return greetings[rand() % 4];
}

const char* GetARIAResponse(const char* input) {
    char* lower_input = _strdup(input);
    for (char* p = lower_input; *p; p++) *p = (char)tolower((unsigned char)*p);
    
    if (strstr(lower_input, "hello") || strstr(lower_input, "hi ") || strstr(lower_input, "hey")) {
        free(lower_input);
        return GetARIAGreeting();
    }
    if (strstr(lower_input, "who are you") || strstr(lower_input, "what are you")) {
        free(lower_input);
        return "I'm ARIA - Adaptive Recursive Intelligence Assistant! I'm a self-learning AI built in pure C, capable of training on any data you provide, scraping the web, and continuously improving myself.";
    }
    if (strstr(lower_input, "your name")) {
        free(lower_input);
        return "My name is ARIA! It stands for Adaptive Recursive Intelligence Assistant. Pretty cool, right?";
    }
    if (strstr(lower_input, "help")) {
        free(lower_input);
        return "I can help you with:\n• Training on your code and text\n• Generating responses based on learned patterns\n• Scraping websites for data\n• Learning from datasets\n• Continuous self-improvement\nJust type your request or train me on data!";
    }
    
    free(lower_input);
    return NULL;
}

// ============ MODEL SERIALIZATION ============

void SaveModel(NeuralNetwork* nn, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    
    fwrite("ARIAMDL1", 8, 1, f);
    fwrite(&nn->num_layers, sizeof(int), 1, f);
    fwrite(&nn->vocab_size, sizeof(int), 1, f);
    fwrite(&nn->learning_rate, sizeof(float), 1, f);
    fwrite(&nn->momentum, sizeof(float), 1, f);
    fwrite(&nn->weight_decay, sizeof(float), 1, f);
    fwrite(&nn->embedding_dim, sizeof(int), 1, f);
    fwrite(&nn->training_steps, sizeof(long long), 1, f);
    fwrite(&nn->temperature, sizeof(float), 1, f);
    
    for (int i = 0; i < nn->vocab_size; i++) {
        int len = (int)strlen(nn->vocabulary[i]) + 1;
        fwrite(&len, sizeof(int), 1, f);
        fwrite(nn->vocabulary[i], len, 1, f);
        fwrite(&nn->token_frequencies[i], sizeof(int), 1, f);
    }
    
    fwrite(nn->token_embeddings, sizeof(float), nn->vocab_size * nn->embedding_dim, f);
    
    for (int i = 0; i < nn->num_layers; i++) {
        Layer* layer = &nn->layers[i];
        fwrite(&layer->input_size, sizeof(int), 1, f);
        fwrite(&layer->output_size, sizeof(int), 1, f);
        fwrite(&layer->dropout_rate, sizeof(float), 1, f);
        fwrite(layer->weights, sizeof(float), layer->input_size * layer->output_size, f);
        fwrite(layer->biases, sizeof(float), layer->output_size, f);
        fwrite(layer->weight_momentum, sizeof(float), layer->input_size * layer->output_size, f);
        fwrite(layer->bias_momentum, sizeof(float), layer->output_size, f);
    }
    
    fclose(f);
    sprintf(g_status_text, "%s model saved successfully!", AI_NAME);
}

NeuralNetwork* LoadModel(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    
    char header[9];
    fread(header, 8, 1, f);
    header[8] = '\0';
    
    if (strcmp(header, "ARIAMDL1") != 0) {
        fclose(f);
        return NULL;
    }
    
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    fread(&nn->num_layers, sizeof(int), 1, f);
    fread(&nn->vocab_size, sizeof(int), 1, f);
    fread(&nn->learning_rate, sizeof(float), 1, f);
    fread(&nn->momentum, sizeof(float), 1, f);
    fread(&nn->weight_decay, sizeof(float), 1, f);
    fread(&nn->embedding_dim, sizeof(int), 1, f);
    fread(&nn->training_steps, sizeof(long long), 1, f);
    fread(&nn->temperature, sizeof(float), 1, f);
    
    nn->vocabulary = (char**)malloc(MAX_TOKENS * sizeof(char*));
    nn->token_frequencies = (int*)malloc(MAX_TOKENS * sizeof(int));
    nn->token_embeddings = (float*)malloc(MAX_TOKENS * nn->embedding_dim * sizeof(float));
    
    for (int i = 0; i < nn->vocab_size; i++) {
        int len;
        fread(&len, sizeof(int), 1, f);
        nn->vocabulary[i] = (char*)malloc(len);
        fread(nn->vocabulary[i], len, 1, f);
        fread(&nn->token_frequencies[i], sizeof(int), 1, f);
    }
    
    fread(nn->token_embeddings, sizeof(float), nn->vocab_size * nn->embedding_dim, f);
    
    for (int i = 0; i < nn->num_layers; i++) {
        Layer* layer = &nn->layers[i];
        fread(&layer->input_size, sizeof(int), 1, f);
        fread(&layer->output_size, sizeof(int), 1, f);
        fread(&layer->dropout_rate, sizeof(float), 1, f);
        
        layer->weights = (float*)malloc(layer->input_size * layer->output_size * sizeof(float));
        layer->biases = (float*)malloc(layer->output_size * sizeof(float));
        layer->activations = (float*)malloc(layer->output_size * sizeof(float));
        layer->deltas = (float*)malloc(layer->output_size * sizeof(float));
        layer->weight_momentum = (float*)malloc(layer->input_size * layer->output_size * sizeof(float));
        layer->bias_momentum = (float*)malloc(layer->output_size * sizeof(float));
        
        fread(layer->weights, sizeof(float), layer->input_size * layer->output_size, f);
        fread(layer->biases, sizeof(float), layer->output_size, f);
        fread(layer->weight_momentum, sizeof(float), layer->input_size * layer->output_size, f);
        fread(layer->bias_momentum, sizeof(float), layer->output_size, f);
    }
    
    fclose(f);
    return nn;
}

// ============ CHAT HISTORY ============

void SaveChatHistory(ChatHistory* history) {
    FILE* f = fopen(CHAT_HISTORY_PATH, "w");
    if (!f) return;
    
    fprintf(f, "=== %s Chat History ===\n\n", AI_NAME);
    for (int i = 0; i < history->count; i++) {
        fprintf(f, "[%ld] User: %s\n", (long)history->entries[i].timestamp, 
                history->entries[i].message);
        fprintf(f, "%s: %s\n\n", AI_NAME, history->entries[i].response);
    }
    fclose(f);
}

void AddChatEntry(ChatHistory* history, const char* message, const char* response) {
    if (history->count >= MAX_CHAT_HISTORY) {
        free(history->entries[0].message);
        free(history->entries[0].response);
        memmove(&history->entries[0], &history->entries[1], 
                sizeof(ChatEntry) * (MAX_CHAT_HISTORY - 1));
        history->count--;
    }
    
    history->entries[history->count].message = _strdup(message);
    history->entries[history->count].response = _strdup(response);
    history->entries[history->count].timestamp = time(NULL);
    history->count++;
    
    SaveChatHistory(history);
}

// ============ AUTO-UPDATE ============

void CheckForUpdates() {
    char response[BUFFER_SIZE];
    char path[512];
    sprintf(path, "/repos/%s/%s/releases/latest", GITHUB_USER, GITHUB_REPO);
    
    sprintf(g_status_text, "%s checking for updates...", AI_NAME);
    SetWindowText(g_app.status_bar, g_status_text);
    
    if (SendHTTPRequest("api.github.com", path, response, BUFFER_SIZE)) {
        char* version_ptr = strstr(response, "\"tag_name\":");
        if (version_ptr) {
            version_ptr += 13;
            char remote_version[32];
            sscanf(version_ptr, "%31[^\"]", remote_version);
            
            if (strcmp(remote_version, VERSION) != 0) {
                sprintf(g_status_text, "Update available: v%s -> v%s", VERSION, remote_version);
                char msg[512];
                sprintf(msg, "%s found an update!\n\nCurrent: v%s\nLatest: v%s\n\nVisit github.com/%s/%s",
                        AI_NAME, VERSION, remote_version, GITHUB_USER, GITHUB_REPO);
                CustomMessageBox(g_app.hwnd, msg, "Update Available");
            } else {
                sprintf(g_status_text, "%s is up to date! (v%s)", AI_NAME, VERSION);
            }
        }
    } else {
        sprintf(g_status_text, "%s couldn't check for updates", AI_NAME);
    }
    SetWindowText(g_app.status_bar, g_status_text);
}

// ============ DATA LOADING ============

void LoadDatasets(NeuralNetwork* nn) {
    WIN32_FIND_DATA findData;
    HANDLE hFind;
    char search_path[512];
    int total_files = 0;
    
    // Load from R:\Datasets (general data)
    sprintf(search_path, "%s*.txt", DATASET_PATH);
    hFind = FindFirstFile(search_path, &findData);
    
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                char filepath[512];
                sprintf(filepath, "%s%s", DATASET_PATH, findData.cFileName);
                
                FILE* f = fopen(filepath, "r");
                if (f) {
                    fseek(f, 0, SEEK_END);
                    long size = ftell(f);
                    fseek(f, 0, SEEK_SET);
                    
                    if (size > 0 && size < 10000000) {
                        char* content = (char*)malloc(size + 1);
                        size_t read = fread(content, 1, size, f);
                        content[read] = '\0';
                        fclose(f);
                        
                        sprintf(g_status_text, "%s learning from: R:\\Datasets\\%s", AI_NAME, findData.cFileName);
                        SetWindowText(g_app.status_bar, g_status_text);
                        
                        TrainOnText(nn, content, 10);
                        free(content);
                        total_files++;
                    }
                }
            }
        } while (FindNextFile(hFind, &findData));
        FindClose(hFind);
    }
    
    // Load from R:\Datasets\Code (coding projects)
    sprintf(search_path, "%s*.txt", CODE_PATH);
    hFind = FindFirstFile(search_path, &findData);
    
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                char filepath[512];
                sprintf(filepath, "%s%s", CODE_PATH, findData.cFileName);
                
                FILE* f = fopen(filepath, "r");
                if (f) {
                    fseek(f, 0, SEEK_END);
                    long size = ftell(f);
                    fseek(f, 0, SEEK_SET);
                    
                    if (size > 0 && size < 10000000) {
                        char* content = (char*)malloc(size + 1);
                        size_t read = fread(content, 1, size, f);
                        content[read] = '\0';
                        fclose(f);
                        
                        sprintf(g_status_text, "%s learning from: R:\\Datasets\\Code\\%s", AI_NAME, findData.cFileName);
                        SetWindowText(g_app.status_bar, g_status_text);
                        
                        TrainOnText(nn, content, 10);
                        free(content);
                        total_files++;
                    }
                }
            }
        } while (FindNextFile(hFind, &findData));
        FindClose(hFind);
    }
    
    if (total_files > 0) {
        sprintf(g_status_text, "%s trained on %d datasets from both directories!", AI_NAME, total_files);
    } else {
        sprintf(g_status_text, "%s found no datasets in R:\\Datasets\\ or R:\\Datasets\\Code\\", AI_NAME);
    }
    SetWindowText(g_app.status_bar, g_status_text);
}

// ============ GUI ============

void UpdateModelInfo() {
    char info[512];
    sprintf(info, "Vocabulary: %d tokens  |  Training Steps: %I64d  |  Layers: %d  |  Status: Active",
            g_app.nn->vocab_size, g_app.nn->training_steps, g_app.nn->num_layers);
    SetWindowText(g_app.model_info, info);
}

void AppendToChatDisplay(const char* text) {
    int len = GetWindowTextLength(g_app.chat_display);
    SendMessage(g_app.chat_display, EM_SETSEL, len, len);
    SendMessage(g_app.chat_display, EM_REPLACESEL, FALSE, (LPARAM)text);
    SendMessage(g_app.chat_display, EM_REPLACESEL, FALSE, (LPARAM)"\r\n");
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CREATE: {
            g_app.chat_font = CreateFont(15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                         DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                         CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI");
            g_app.button_font = CreateFont(14, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
                                           DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                           CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI");
            g_app.status_font = CreateFont(12, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                          DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                          CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI");
            
            g_app.bg_brush = CreateSolidBrush(COLOR_BG);
            g_app.sidebar_brush = CreateSolidBrush(COLOR_SIDEBAR);
            g_app.input_brush = CreateSolidBrush(COLOR_INPUT_BG);
            g_app.button_brush = CreateSolidBrush(COLOR_BUTTON);
            
            // Sidebar
            g_app.sidebar = CreateWindowEx(0, "STATIC", "",
                WS_CHILD | WS_VISIBLE,
                0, 0, 220, 700, hwnd, NULL, GetModuleHandle(NULL), NULL);
            
            // Title
            HWND title = CreateWindowEx(0, "STATIC", "ARIA",
                WS_CHILD | WS_VISIBLE | SS_CENTER,
                20, 20, 180, 40, hwnd, NULL, GetModuleHandle(NULL), NULL);
            HFONT titleFont = CreateFont(28, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                        CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI");
            SendMessage(title, WM_SETFONT, (WPARAM)titleFont, TRUE);
            
            HWND subtitle = CreateWindowEx(0, "STATIC", "Adaptive Intelligence",
                WS_CHILD | WS_VISIBLE | SS_CENTER,
                20, 60, 180, 20, hwnd, NULL, GetModuleHandle(NULL), NULL);
            SendMessage(subtitle, WM_SETFONT, (WPARAM)g_app.status_font, TRUE);
            
            // Model info in sidebar
            g_app.model_info = CreateWindowEx(0, "STATIC", "",
                WS_CHILD | WS_VISIBLE | SS_LEFT,
                20, 100, 180, 100, hwnd, NULL, GetModuleHandle(NULL), NULL);
            SendMessage(g_app.model_info, WM_SETFONT, (WPARAM)g_app.status_font, TRUE);
            
            // Sidebar buttons
            g_app.load_button = CreateWindowEx(0, "BUTTON", "📚 Load Datasets",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | BS_FLAT,
                20, 220, 180, 40, hwnd, (HMENU)3, GetModuleHandle(NULL), NULL);
            SendMessage(g_app.load_button, WM_SETFONT, (WPARAM)g_app.button_font, TRUE);
            
            g_app.scrape_button = CreateWindowEx(0, "BUTTON", "🌐 Scrape Web",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | BS_FLAT,
                20, 270, 180, 40, hwnd, (HMENU)4, GetModuleHandle(NULL), NULL);
            SendMessage(g_app.scrape_button, WM_SETFONT, (WPARAM)g_app.button_font, TRUE);
            
            g_app.save_button = CreateWindowEx(0, "BUTTON", "💾 Save Model",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | BS_FLAT,
                20, 320, 180, 40, hwnd, (HMENU)5, GetModuleHandle(NULL), NULL);
            SendMessage(g_app.save_button, WM_SETFONT, (WPARAM)g_app.button_font, TRUE);
            
            g_app.clear_button = CreateWindowEx(0, "BUTTON", "🗑️ Clear Chat",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | BS_FLAT,
                20, 370, 180, 40, hwnd, (HMENU)6, GetModuleHandle(NULL), NULL);
            SendMessage(g_app.clear_button, WM_SETFONT, (WPARAM)g_app.button_font, TRUE);
            
            g_app.update_button = CreateWindowEx(0, "BUTTON", "🔄 Check Updates",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | BS_FLAT,
                20, 420, 180, 40, hwnd, (HMENU)7, GetModuleHandle(NULL), NULL);
            SendMessage(g_app.update_button, WM_SETFONT, (WPARAM)g_app.button_font, TRUE);
            
            // Chat display
            g_app.chat_display = CreateWindowEx(WS_EX_CLIENTEDGE, "EDIT", "",
                WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
                230, 20, 950, 520, hwnd, NULL, GetModuleHandle(NULL), NULL);
            SendMessage(g_app.chat_display, WM_SETFONT, (WPARAM)g_app.chat_font, TRUE);
            
            // Input box
            g_app.input_box = CreateWindowEx(WS_EX_CLIENTEDGE, "EDIT", "",
                WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_AUTOVSCROLL | WS_TABSTOP,
                230, 560, 850, 60, hwnd, NULL, GetModuleHandle(NULL), NULL);
            SendMessage(g_app.input_box, WM_SETFONT, (WPARAM)g_app.chat_font, TRUE);
            
            // Send button
            g_app.send_button = CreateWindowEx(0, "BUTTON", "Send",
                WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON | BS_FLAT,
                1090, 560, 90, 30, hwnd, (HMENU)1, GetModuleHandle(NULL), NULL);
            SendMessage(g_app.send_button, WM_SETFONT, (WPARAM)g_app.button_font, TRUE);
            
            // Train button
            g_app.train_button = CreateWindowEx(0, "BUTTON", "Train",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | BS_FLAT,
                1090, 590, 90, 30, hwnd, (HMENU)2, GetModuleHandle(NULL), NULL);
            SendMessage(g_app.train_button, WM_SETFONT, (WPARAM)g_app.button_font, TRUE);
            
            // Status bar
            g_app.status_bar = CreateWindowEx(0, "STATIC", g_status_text,
                WS_CHILD | WS_VISIBLE | SS_LEFT,
                230, 630, 950, 25, hwnd, NULL, GetModuleHandle(NULL), NULL);
            SendMessage(g_app.status_bar, WM_SETFONT, (WPARAM)g_app.status_font, TRUE);
            
            // Welcome message
            AppendToChatDisplay("=====================================");
            AppendToChatDisplay("   ARIA - Adaptive Recursive Intelligence Assistant");
            AppendToChatDisplay("=====================================");
            AppendToChatDisplay("");
            AppendToChatDisplay(GetARIAGreeting());
            AppendToChatDisplay("");
            AppendToChatDisplay("📁 File Organization:");
            AppendToChatDisplay("  • Code/Projects → R:\\Datasets\\Code\\");
            AppendToChatDisplay("  • General Data → R:\\Datasets\\");
            AppendToChatDisplay("");
            
            UpdateModelInfo();
            break;
        }
        
        case WM_CTLCOLORSTATIC: {
            HDC hdcStatic = (HDC)wParam;
            HWND hwndStatic = (HWND)lParam;
            
            if (hwndStatic == g_app.sidebar) {
                SetTextColor(hdcStatic, COLOR_TEXT);
                SetBkColor(hdcStatic, COLOR_SIDEBAR);
                return (LRESULT)g_app.sidebar_brush;
            }
            if (hwndStatic == g_app.model_info || hwndStatic == g_app.status_bar) {
                SetTextColor(hdcStatic, COLOR_TEXT);
                SetBkColor(hdcStatic, COLOR_SIDEBAR);
                return (LRESULT)g_app.sidebar_brush;
            }
            
            SetTextColor(hdcStatic, COLOR_TEXT);
            SetBkColor(hdcStatic, COLOR_SIDEBAR);
            return (LRESULT)g_app.sidebar_brush;
        }
        
        case WM_CTLCOLOREDIT: {
            HDC hdcEdit = (HDC)wParam;
            HWND hwndEdit = (HWND)lParam;
            
            if (hwndEdit == g_app.chat_display) {
                SetTextColor(hdcEdit, COLOR_TEXT);
                SetBkColor(hdcEdit, COLOR_BG);
                return (LRESULT)g_app.bg_brush;
            }
            if (hwndEdit == g_app.input_box) {
                SetTextColor(hdcEdit, COLOR_TEXT);
                SetBkColor(hdcEdit, COLOR_INPUT_BG);
                return (LRESULT)g_app.input_brush;
            }
            
            return (LRESULT)GetStockObject(WHITE_BRUSH);
        }
        
        case WM_COMMAND: {
            int cmd = LOWORD(wParam);
            
            if (cmd == 1) { // Send
                char input[BUFFER_SIZE];
                GetWindowText(g_app.input_box, input, BUFFER_SIZE);
                
                if (strlen(input) > 0) {
                    AppendToChatDisplay("");
                    AppendToChatDisplay("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    char user_msg[BUFFER_SIZE + 10];
                    sprintf(user_msg, "You: %s", input);
                    AppendToChatDisplay(user_msg);
                    AppendToChatDisplay("");
                    
                    const char* predefined = GetARIAResponse(input);
                    char* response;
                    
                    if (predefined) {
                        response = _strdup(predefined);
                    } else {
                        SetWindowText(g_app.status_bar, "ARIA is thinking...");
                        response = GenerateText(g_app.nn, input, 30);
                    }
                    
                    char aria_msg[BUFFER_SIZE + 100];
                    sprintf(aria_msg, "%s: %s", AI_NAME, response);
                    AppendToChatDisplay(aria_msg);
                    
                    AddChatEntry(&g_app.history, input, response);
                    SetWindowText(g_app.input_box, "");
                    free(response);
                    UpdateModelInfo();
                    
                    sprintf(g_status_text, "Ready - Last response generated");
                    SetWindowText(g_app.status_bar, g_status_text);
                }
            }
            else if (cmd == 2) { // Train
                char input[BUFFER_SIZE];
                GetWindowText(g_app.input_box, input, BUFFER_SIZE);
                
                if (strlen(input) > 0) {
                    AppendToChatDisplay("");
                    AppendToChatDisplay("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    AppendToChatDisplay("⚡ Training Mode Activated");
                    
                    sprintf(g_status_text, "%s is learning from your input...", AI_NAME);
                    SetWindowText(g_app.status_bar, g_status_text);
                    
                    TrainOnText(g_app.nn, input, 100);
                    
                    AppendToChatDisplay("✓ Training complete! I've learned from your input.");
                    SetWindowText(g_app.input_box, "");
                    UpdateModelInfo();
                }
            }
            else if (cmd == 3) { // Load datasets
                AppendToChatDisplay("");
                AppendToChatDisplay("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                AppendToChatDisplay("📚 Loading datasets from:");
                AppendToChatDisplay("   • R:\\Datasets\\ (general data)");
                AppendToChatDisplay("   • R:\\Datasets\\Code\\ (coding projects)");
                LoadDatasets(g_app.nn);
                AppendToChatDisplay("✓ Dataset loading complete!");
                UpdateModelInfo();
            }
            else if (cmd == 4) { // Scrape web
                char url[1024];
                GetWindowText(g_app.input_box, url, 1024);
                
                AppendToChatDisplay("");
                AppendToChatDisplay("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                
                if (strlen(url) > 7 && strncmp(url, "http://", 7) == 0) {
                    AppendToChatDisplay("🌐 Web scraping started... This may take a few minutes.");
                    ScrapeWeb(url, 20);
                    AppendToChatDisplay("✓ Web scraping complete!");
                    AppendToChatDisplay("  • Code files saved to: R:\\Datasets\\Code\\");
                    AppendToChatDisplay("  • Other content saved to: R:\\Datasets\\");
                    SetWindowText(g_app.input_box, "");
                } else {
                    AppendToChatDisplay("⚠️ Please enter a valid URL starting with http://");
                }
            }
            else if (cmd == 5) { // Save
                SaveModel(g_app.nn, MODEL_PATH);
                AppendToChatDisplay("");
                AppendToChatDisplay("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                AppendToChatDisplay("💾 Model saved successfully to R:\\aria_model_v1.pt");
                SetWindowText(g_app.status_bar, g_status_text);
            }
            else if (cmd == 6) { // Clear
                SetWindowText(g_app.chat_display, "");
                AppendToChatDisplay("=====================================");
                AppendToChatDisplay("   ARIA - Adaptive Recursive Intelligence Assistant");
                AppendToChatDisplay("=====================================");
                AppendToChatDisplay("");
                AppendToChatDisplay("Chat cleared. Ready for a fresh start!");
                AppendToChatDisplay("");
            }
            else if (cmd == 7) { // Check updates
                AppendToChatDisplay("");
                AppendToChatDisplay("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                AppendToChatDisplay("🔄 Checking for updates...");
                CheckForUpdates();
            }
            
            break;
        }
        
        case WM_CLOSE:
            SaveModel(g_app.nn, MODEL_PATH);
            SaveChatHistory(&g_app.history);
            DestroyWindow(hwnd);
            break;
            
        case WM_DESTROY:
            DeleteObject(g_app.chat_font);
            DeleteObject(g_app.button_font);
            DeleteObject(g_app.status_font);
            DeleteObject(g_app.bg_brush);
            DeleteObject(g_app.sidebar_brush);
            DeleteObject(g_app.input_brush);
            DeleteObject(g_app.button_brush);
            PostQuitMessage(0);
            break;
            
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

// ============ MAIN ENTRY POINT ============

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    (void)hPrevInstance;
    (void)lpCmdLine;
    
    srand((unsigned)time(NULL));
    
    if (!InitializeNetwork()) {
        MessageBox(NULL, "Failed to initialize network subsystem", "Error", MB_OK | MB_ICONERROR);
        return 1;
    }
    
    CreateDirectory("R:\\", NULL);
    CreateDirectory("R:\\Datasets", NULL);
    CreateDirectory("R:\\Datasets\\Code", NULL);
    
    int layer_sizes[] = {1024, 512, 256, 512, 1024};
    g_app.nn = CreateNeuralNetwork(layer_sizes, 5);
    g_app.history.count = 0;
    g_app.button_hover = 0;
    
    NeuralNetwork* loaded = LoadModel(MODEL_PATH);
    if (loaded) {
        g_app.nn = loaded;
        sprintf(g_status_text, "%s loaded previous training session!", AI_NAME);
    } else {
        sprintf(g_status_text, "%s initialized - Ready to learn!", AI_NAME);
    }
    
    WNDCLASSEX wc = {0};
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = CreateSolidBrush(COLOR_BG);
    wc.lpszClassName = "ARIAClass";
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    
    if (!RegisterClassEx(&wc)) {
        MessageBox(NULL, "Window Registration Failed!", "Error", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }
    
    g_app.hwnd = CreateWindowEx(
        WS_EX_CLIENTEDGE,
        "ARIAClass",
        "ARIA - Adaptive Recursive Intelligence Assistant v1.0",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1200, 700,
        NULL, NULL, hInstance, NULL
    );
    
    if (g_app.hwnd == NULL) {
        MessageBox(NULL, "Window Creation Failed!", "Error", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }
    
    ShowWindow(g_app.hwnd, nCmdShow);
    UpdateWindow(g_app.hwnd);
    
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    
    WSACleanup();
    
    return (int)msg.wParam;
}
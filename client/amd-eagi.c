/*
 * amd-eagi: Asterisk (13+) EAGI client for the acoustic AMD server.
 *
 * Reads the answered call's audio from EAGI fd 3 (slin16, 8 kHz, mono), streams it to
 * the AMD server over a WebSocket in 100 ms binary frames, waits for the verdict and
 * sets two channel variables:
 *
 *   AMDSTATUS = HUMAN | MACHINE
 *   AMDCAUSE  = HUMAN | MACHINE | SCREENING | SILENCE | TONE | BUSY | UNKNOWN
 *
 * Any error or timeout sets MACHINE / UNKNOWN (only a confirmed HUMAN is passed as HUMAN).
 * No dependencies; one process per call, ~1 MB RSS.
 *
 * Dialplan:
 *   same => n,EAGI(amd-eagi,ws://10.0.0.5:8085/,3000)
 *   same => n,NoOp(AMD ${AMDSTATUS} ${AMDCAUSE})
 * Arguments: server URL (ws://IP:port/path, an IP address: no DNS lookup), and an
 * optional overall timeout in ms (default 3000).
 *
 * Build: gcc -O2 -static -o amd-eagi amd-eagi.c
 */
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define AUDIO_FD 3
#define FRAME_BYTES 1600   /* 100 ms of slin16 @ 8 kHz */
#define WINDOW_BYTES 32000 /* 2.0 s: the server decides once it has this much */

static char status[16] = "MACHINE", cause[16] = "UNKNOWN";

static long now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000L + t.tv_nsec / 1000000L;
}

/* AGI: send a command, read (and ignore) the one-line response */
static void agi(const char *fmt, const char *a, const char *b) {
    char line[512];
    printf(fmt, a, b);
    printf("\n");
    fflush(stdout);
    if (!fgets(line, sizeof line, stdin)) return;
}

static void finish(void) {
    agi("SET VARIABLE AMDSTATUS \"%s\"%s", status, "");
    agi("SET VARIABLE AMDCAUSE \"%s\"%s", cause, "");
    agi("VERBOSE \"amd-eagi: %s %s\" 3", status, cause);
}

static int send_all(int fd, const unsigned char *p, size_t n) {
    while (n) {
        ssize_t w = send(fd, p, n, MSG_NOSIGNAL);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        p += w; n -= (size_t)w;
    }
    return 0;
}

/* client -> server frames must be masked (RFC 6455) */
static int ws_send(int fd, int opcode, const unsigned char *data, size_t n) {
    unsigned char hdr[14], mask[4];
    size_t h = 0;
    unsigned char buf[FRAME_BYTES + 256];
    hdr[h++] = 0x80 | opcode;
    if (n < 126) hdr[h++] = 0x80 | n;
    else { hdr[h++] = 0x80 | 126; hdr[h++] = n >> 8; hdr[h++] = n & 0xff; }
    for (int i = 0; i < 4; i++) mask[i] = rand() & 0xff;
    memcpy(hdr + h, mask, 4); h += 4;
    if (n > sizeof buf) return -1;
    for (size_t i = 0; i < n; i++) buf[i] = data[i] ^ mask[i & 3];
    if (send_all(fd, hdr, h)) return -1;
    return send_all(fd, buf, n);
}

/* pull a "key":"VALUE" string out of the verdict JSON */
static void json_str(const char *js, const char *key, char *out, size_t outn) {
    char pat[32];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(js, pat);
    if (!p) return;
    p = strchr(p + strlen(pat), '"');
    if (!p) return;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < outn) out[i++] = *p++;
    out[i] = 0;
}

static int parse_url(const char *url, char *host, size_t hn, int *port, char *path, size_t pn) {
    if (strncmp(url, "ws://", 5)) return -1;
    const char *h = url + 5, *c = strchr(h, ':'), *s = strchr(h, '/');
    if (!s) s = h + strlen(h);
    *port = 80;
    const char *hend = s;
    if (c && c < s) { *port = atoi(c + 1); hend = c; }
    if ((size_t)(hend - h) >= hn) return -1;
    memcpy(host, h, hend - h); host[hend - h] = 0;
    snprintf(path, pn, "%s", *s ? s : "/");
    return 0;
}

int main(int argc, char **argv) {
    char line[1024], uniqueid[96] = "";
    /* AGI environment: "agi_xxx: value" lines until a blank line */
    while (fgets(line, sizeof line, stdin) && line[0] != '\n') {
        if (!strncmp(line, "agi_uniqueid: ", 14)) {
            snprintf(uniqueid, sizeof uniqueid, "%.95s", line + 14);
            uniqueid[strcspn(uniqueid, "\r\n")] = 0;
        }
    }
    if (argc < 2) { finish(); return 0; }
    long timeout = argc > 2 ? atol(argv[2]) : 3000;
    long t0 = now_ms();
    srand((unsigned)(t0 ^ getpid()));

    char host[64], path[256];
    int port;
    if (parse_url(argv[1], host, sizeof host, &port, path, sizeof path)) { finish(); return 0; }
    struct sockaddr_in sa = {0};
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) { finish(); return 0; }

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct timeval tv = {1, 0};
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    if (fd < 0 || connect(fd, (struct sockaddr *)&sa, sizeof sa)) { finish(); return 0; }

    /* WebSocket handshake */
    char req[512];
    int rn = snprintf(req, sizeof req,
                      "GET %s HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n",
                      path, host, port);
    if (send_all(fd, (unsigned char *)req, rn)) { finish(); return 0; }
    char resp[1024] = "";
    size_t got = 0;
    while (got < sizeof resp - 1 && !strstr(resp, "\r\n\r\n")) {
        struct pollfd p = {fd, POLLIN, 0};
        long left = timeout - (now_ms() - t0);
        if (left <= 0 || poll(&p, 1, left) <= 0) { finish(); return 0; }
        ssize_t r = recv(fd, resp + got, 1, 0); /* byte at a time: never read past the headers */
        if (r <= 0) { finish(); return 0; }
        got += r; resp[got] = 0;
    }
    if (!strstr(resp, " 101")) { finish(); return 0; }

    char start[160];
    int sn = snprintf(start, sizeof start, "{\"type\":\"start\",\"call_id\":\"%s\"}", uniqueid);
    ws_send(fd, 0x1, (unsigned char *)start, sn);

    unsigned char audio[FRAME_BYTES];
    size_t have = 0, sent = 0;
    int audio_open = 1;
    unsigned char in[4096];
    size_t inlen = 0;
    for (;;) {
        long left = timeout - (now_ms() - t0);
        if (left <= 0) break;
        struct pollfd p[2] = {{fd, POLLIN, 0}, {AUDIO_FD, POLLIN, 0}};
        int np = (audio_open && sent < WINDOW_BYTES) ? 2 : 1;
        if (poll(p, np, left) <= 0) break;
        if (np == 2 && (p[1].revents & (POLLIN | POLLHUP))) {
            ssize_t r = read(AUDIO_FD, audio + have, FRAME_BYTES - have);
            if (r <= 0) { /* call ended: let the server decide with what it has */
                audio_open = 0;
                if (have) ws_send(fd, 0x2, audio, have);
                ws_send(fd, 0x1, (unsigned char *)"{\"type\":\"close\"}", 16);
            } else {
                have += r;
                if (have == FRAME_BYTES) {
                    if (ws_send(fd, 0x2, audio, have)) break;
                    sent += have; have = 0;
                }
            }
        }
        if (p[0].revents & (POLLIN | POLLHUP)) {
            ssize_t r = recv(fd, in + inlen, sizeof in - 1 - inlen, 0);
            if (r <= 0) break;
            inlen += r;
            /* server frames are unmasked; the verdict is one small text frame */
            if (inlen >= 2) {
                size_t len = in[1] & 0x7f, off = 2;
                if (len == 126) { if (inlen < 4) continue; len = (in[2] << 8) | in[3]; off = 4; }
                if (inlen < off + len) continue;
                if ((in[0] & 0x0f) == 0x1) {
                    in[off + len] = 0;
                    json_str((char *)in + off, "amdstatus", status, sizeof status);
                    json_str((char *)in + off, "amdcause", cause, sizeof cause);
                    if (strcmp(status, "HUMAN") && strcmp(status, "MACHINE")) {
                        strcpy(status, "MACHINE"); strcpy(cause, "UNKNOWN");
                    }
                    if (!strcmp(cause, "HUMAN") != !strcmp(status, "HUMAN")) {  /* enforce the rule */
                        strcpy(status, "MACHINE");
                    }
                }
                break;
            }
        }
    }
    unsigned char bye[2] = {0x03, 0xe8};
    ws_send(fd, 0x8, bye, 2);
    close(fd);
    finish();
    return 0;
}

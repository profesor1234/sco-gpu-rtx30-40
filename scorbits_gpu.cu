/*
 * Scorbits (SCO) GPU Miner v7 — RTX 30/40 + Linux (vast.ai)
 *
 * Optimizations for Ampere (sm_86) and Ada Lovelace (sm_89):
 *   - K constants in __constant__ memory (works correctly on sm_86/89, NOT sm_61)
 *   - launch_bounds(256,8) — RTX 30/40 have 64K regs/SM (2x sm_61)
 *   - SM*64 grid — RTX 3090=82 SMs, RTX 4090=128 SMs
 *   - Multi-GPU: --gpu flag or auto-detect all GPUs
 *   - SHA-256 midstate precomputation (CPU hashes first 64B block)
 *   - 3s difficulty change polling (matches official miner v4)
 *   - Smart anti-spike recovery (re-mine on rejection)
 *   - POSIX sockets for Linux (no winsock)
 *
 * Build (vast.ai Ubuntu):
 *   nvcc -O3 \
 *     -gencode arch=compute_86,code=sm_86 \
 *     -gencode arch=compute_89,code=sm_89 \
 *     -o scorbits_gpu scorbits_gpu.cu
 *
 * Or for a specific GPU:
 *   nvcc -O3 -arch=sm_86 -o scorbits_gpu scorbits_gpu.cu   # RTX 30xx
 *   nvcc -O3 -arch=sm_89 -o scorbits_gpu scorbits_gpu.cu   # RTX 40xx
 *
 * Run:
 *   ./scorbits_gpu --address SCO... --node http://51.91.122.48:8080
 *   ./scorbits_gpu --address SCO... --node http://51.91.122.48:8080 --gpu 0
 *   ./scorbits_gpu --address SCO... --node http://51.91.122.48:8080 --gpu 1
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>
#include <sys/time.h>

/* ── SHA-256 macros ──────────────────────────────────────────────── */
#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z)   (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z)  (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x)      (ROTR32(x,2)^ROTR32(x,13)^ROTR32(x,22))
#define EP1(x)      (ROTR32(x,6)^ROTR32(x,11)^ROTR32(x,25))
#define SIG0(x)     (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define SIG1(x)     (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))

/* __constant__ K — broadcast via constant cache, works on sm_86+ */
__constant__ uint32_t K_GPU[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

static const uint32_t K_CPU[64]={
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

/* ── Midstate ────────────────────────────────────────────────────── */
typedef struct { uint32_t h[8]; char tail[256]; int tail_len; int total_prefix_len; } Midstate;

static void sha256_block_cpu(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    for(int i=0;i<16;i++) w[i]=((uint32_t)block[i*4]<<24)|((uint32_t)block[i*4+1]<<16)|((uint32_t)block[i*4+2]<<8)|block[i*4+3];
    for(int i=16;i<64;i++) w[i]=SIG1(w[i-2])+w[i-7]+SIG0(w[i-15])+w[i-16];
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    for(int i=0;i<64;i++){uint32_t t1=h+EP1(e)+CH(e,f,g)+K_CPU[i]+w[i];uint32_t t2=EP0(a)+MAJ(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

static void compute_midstate(const char* prefix, int plen, Midstate* ms) {
    ms->h[0]=0x6a09e667u;ms->h[1]=0xbb67ae85u;ms->h[2]=0x3c6ef372u;ms->h[3]=0xa54ff53au;
    ms->h[4]=0x510e527fu;ms->h[5]=0x9b05688cu;ms->h[6]=0x1f83d9abu;ms->h[7]=0x5be0cd19u;
    ms->total_prefix_len=plen;
    int full_blocks=plen/64;
    for(int b=0;b<full_blocks;b++) sha256_block_cpu(ms->h,(const uint8_t*)prefix+b*64);
    ms->tail_len=plen-full_blocks*64;
    if(ms->tail_len>0) memcpy(ms->tail,prefix+full_blocks*64,ms->tail_len);
}

/* ── GPU SHA-256 compress — uses __constant__ K_GPU ──────────────── */
__device__ void sha256_compress(uint32_t state[8], const uint32_t w16[16]) {
    uint32_t w[64];
    for(int i=0;i<16;i++) w[i]=w16[i];
    for(int i=16;i<64;i++) w[i]=SIG1(w[i-2])+w[i-7]+SIG0(w[i-15])+w[i-16];
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    for(int i=0;i<64;i++){uint32_t t1=h+EP1(e)+CH(e,f,g)+K_GPU[i]+w[i];uint32_t t2=EP0(a)+MAJ(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

__device__ int gpu_itoa(char* buf, long long v) {
    if(v==0){buf[0]='0';return 1;}
    char tmp[20];int i=0;long long u=v;
    while(u>0){tmp[i++]='0'+(int)(u%10);u/=10;}
    for(int j=0;j<i;j++) buf[j]=tmp[i-1-j];
    return i;
}

/* ── Mining kernel — tuned for RTX 30/40 ─────────────────────────── */
__global__ __launch_bounds__(256, 8) void mine_kernel(
    const uint32_t* __restrict__ d_midstate, const char* __restrict__ d_tail, int tail_len,
    const char* __restrict__ d_suffix, int slen, int prefix_total_len,
    long long base_nonce, int difficulty,
    long long* found_nonce, long long* found_ts, long long current_ts, char* found_hash_hex)
{
    extern __shared__ char smem[];
    char* s_tail=smem; char* s_suffix=smem+tail_len;
    int tid=threadIdx.x;
    for(int i=tid;i<tail_len;i+=blockDim.x) s_tail[i]=d_tail[i];
    for(int i=tid;i<slen;i+=blockDim.x) s_suffix[i]=d_suffix[i];
    __syncthreads();

    long long nonce=base_nonce+(long long)(blockIdx.x*(int)blockDim.x+tid);

    uint8_t data[128]; int pos=0;
    for(int i=0;i<tail_len;i++) data[pos++]=(uint8_t)s_tail[i];
    char ns[20]; int nl=gpu_itoa(ns,nonce);
    for(int i=0;i<nl;i++) data[pos++]=(uint8_t)ns[i];
    for(int i=0;i<slen;i++) data[pos++]=(uint8_t)s_suffix[i];

    uint32_t total_len=(uint32_t)(prefix_total_len+nl+slen);
    data[pos]=0x80;
    int padded_len=((pos+1+8)+63)&~63;
    for(int i=pos+1;i<padded_len;i++) data[i]=0;
    uint64_t bits=(uint64_t)total_len*8ULL;
    data[padded_len-1]=(uint8_t)(bits);     data[padded_len-2]=(uint8_t)(bits>>8);
    data[padded_len-3]=(uint8_t)(bits>>16);  data[padded_len-4]=(uint8_t)(bits>>24);
    data[padded_len-5]=(uint8_t)(bits>>32);  data[padded_len-6]=(uint8_t)(bits>>40);
    data[padded_len-7]=(uint8_t)(bits>>48);  data[padded_len-8]=(uint8_t)(bits>>56);

    uint32_t state[8];
    state[0]=d_midstate[0];state[1]=d_midstate[1];state[2]=d_midstate[2];state[3]=d_midstate[3];
    state[4]=d_midstate[4];state[5]=d_midstate[5];state[6]=d_midstate[6];state[7]=d_midstate[7];

    int rem_blocks=padded_len/64;
    for(int b=0;b<rem_blocks;b++){
        uint32_t w16[16]; uint8_t* bp=data+b*64;
        for(int i=0;i<16;i++) w16[i]=((uint32_t)bp[i*4]<<24)|((uint32_t)bp[i*4+1]<<16)|((uint32_t)bp[i*4+2]<<8)|bp[i*4+3];
        sha256_compress(state,w16);
    }

    /* Early exit cascade */
    if(difficulty>=4 && (state[0]>>16)!=0) return;
    if(difficulty>=2 && (state[0]>>24)!=0) return;

    uint8_t hash[32];
    for(int i=0;i<8;i++){uint32_t v=state[i];hash[i*4]=(v>>24)&0xFF;hash[i*4+1]=(v>>16)&0xFF;hash[i*4+2]=(v>>8)&0xFF;hash[i*4+3]=v&0xFF;}
    int full=difficulty/2; bool ok=true;
    for(int i=0;i<full&&ok;i++) if(hash[i]!=0x00u) ok=false;
    if(ok&&(difficulty&1)) if((hash[full]>>4)!=0) ok=false;

    if(ok){
        unsigned long long prev=atomicCAS((unsigned long long*)found_nonce,(unsigned long long)(-1LL),(unsigned long long)nonce);
        if(prev==(unsigned long long)(-1LL)){
            *found_ts=current_ts;
            const char hx[]="0123456789abcdef";
            for(int i=0;i<32;i++){found_hash_hex[i*2]=hx[hash[i]>>4];found_hash_hex[i*2+1]=hx[hash[i]&0xF];}
            found_hash_hex[64]='\0';
        }
    }
}

/* ── CPU SHA-256 (verification) ──────────────────────────────────── */
static void sha256_host(const uint8_t* data, uint32_t len, uint8_t out[32]) {
    uint32_t h0=0x6a09e667,h1=0xbb67ae85,h2=0x3c6ef372,h3=0xa54ff53a;
    uint32_t h4=0x510e527f,h5=0x9b05688c,h6=0x1f83d9ab,h7=0x5be0cd19;
    uint32_t nb=(len+9+63)/64; uint8_t* padded=(uint8_t*)calloc(nb*64,1);
    memcpy(padded,data,len); padded[len]=0x80;
    uint64_t bits=(uint64_t)len*8;
    for(int i=0;i<8;i++) padded[nb*64-1-i]=(uint8_t)(bits>>(i*8));
    for(uint32_t bn=0;bn<nb;bn++){
        uint8_t* blk=padded+bn*64; uint32_t w[64];
        for(int i=0;i<16;i++) w[i]=((uint32_t)blk[i*4]<<24)|((uint32_t)blk[i*4+1]<<16)|((uint32_t)blk[i*4+2]<<8)|blk[i*4+3];
        for(int i=16;i<64;i++) w[i]=SIG1(w[i-2])+w[i-7]+SIG0(w[i-15])+w[i-16];
        uint32_t a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,hh=h7;
        for(int i=0;i<64;i++){uint32_t t1=hh+EP1(e)+CH(e,f,g)+K_CPU[i]+w[i];uint32_t t2=EP0(a)+MAJ(a,b,c);hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
        h0+=a;h1+=b;h2+=c;h3+=d;h4+=e;h5+=f;h6+=g;h7+=hh;
    }
    free(padded);
    uint32_t hh[8]={h0,h1,h2,h3,h4,h5,h6,h7};
    for(int i=0;i<8;i++){out[i*4]=(hh[i]>>24)&0xFF;out[i*4+1]=(hh[i]>>16)&0xFF;out[i*4+2]=(hh[i]>>8)&0xFF;out[i*4+3]=hh[i]&0xFF;}
}

static void sha256_cpu(const char* input, char* hex_out) {
    uint8_t out[32]; sha256_host((const uint8_t*)input,(uint32_t)strlen(input),out);
    const char hx[]="0123456789abcdef";
    for(int i=0;i<32;i++){hex_out[i*2]=hx[out[i]>>4];hex_out[i*2+1]=hx[out[i]&0xF];}
    hex_out[64]='\0';
}

/* ── POSIX HTTP client (replaces winsock) ────────────────────────── */
static int http_connect_host(const char* host, int port) {
    int s=socket(AF_INET,SOCK_STREAM,0);
    if(s<0) return -1;
    struct timeval tv={30,0};
    setsockopt(s,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof(tv));
    setsockopt(s,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof(tv));
    struct sockaddr_in addr; memset(&addr,0,sizeof(addr));
    addr.sin_family=AF_INET; addr.sin_port=htons(port);
    if(inet_pton(AF_INET,host,&addr.sin_addr)<=0){
        struct hostent* he=gethostbyname(host);
        if(!he){close(s);return -1;}
        memcpy(&addr.sin_addr,he->h_addr,he->h_length);
    }
    if(connect(s,(struct sockaddr*)&addr,sizeof(addr))!=0){close(s);return -1;}
    return s;
}

static int http_request(const char* method, const char* host, int port,
                        const char* path, const char* body,
                        char* out, int outsz, int* status_code) {
    int s=http_connect_host(host,port);
    if(s<0) return -1;
    char req[32768]; int rlen;
    if(body&&strlen(body)>0)
        rlen=sprintf(req,"%s %s HTTP/1.0\r\nHost: %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s",
            method,path,host,(int)strlen(body),body);
    else
        rlen=sprintf(req,"%s %s HTTP/1.0\r\nHost: %s\r\nConnection: close\r\n\r\n",method,path,host);
    send(s,req,rlen,0);
    int n=0,chunk; char tmp[16384];
    while((chunk=recv(s,tmp,sizeof(tmp)-1,0))>0&&n<outsz-1){int cp=chunk<outsz-1-n?chunk:outsz-1-n;memcpy(out+n,tmp,cp);n+=cp;}
    if(n>0)out[n]='\0'; else out[0]='\0';
    close(s);
    if(status_code){*status_code=0;if(strncmp(out,"HTTP/",5)==0)sscanf(out+9,"%d",status_code);}
    char* bs=strstr(out,"\r\n\r\n");
    if(bs) memmove(out,bs+4,strlen(bs+4)+1);
    return n;
}

/* ── JSON helpers ────────────────────────────────────────────────── */
static int jstr(const char* js,const char* key,char* out,int sz){
    char pat[128];sprintf(pat,"\"%s\":",key);const char* p=strstr(js,pat);if(!p)return 0;
    p+=strlen(pat);while(*p==' ')p++;int i=0;
    if(*p=='"'){p++;while(*p&&*p!='"'&&i<sz-1)out[i++]=*p++;}
    else{while(*p&&((*p>='0'&&*p<='9')||*p=='-'||*p=='.')&&i<sz-1)out[i++]=*p++;}
    out[i]='\0';return i>0;
}
static int jbool(const char* js,const char* key){
    char pat[128];sprintf(pat,"\"%s\":",key);const char* p=strstr(js,pat);if(!p)return -1;
    p+=strlen(pat);while(*p==' ')p++;
    if(strncmp(p,"true",4)==0)return 1;if(strncmp(p,"false",5)==0)return 0;return -1;
}

/* ── Work template ───────────────────────────────────────────────── */
typedef struct{int block_index;char previous_hash[256];int difficulty;int reward;long long timestamp;long long last_timestamp;char transactions[2048];}WorkTemplate;

static int fetch_work(const char* host,int port,const char* path,const char* address,WorkTemplate* work){
    static char resp[8192];memset(resp,0,sizeof(resp));int status=0;
    http_request("GET",host,port,path,NULL,resp,sizeof(resp),&status);
    if(status!=200&&status!=0)return 0;if(!resp[0])return 0;char val[256];
    if(!jstr(resp,"block_index",val,sizeof(val)))return 0;work->block_index=atoi(val);
    jstr(resp,"previous_hash",work->previous_hash,sizeof(work->previous_hash));
    if(!jstr(resp,"difficulty",val,sizeof(val)))return 0;work->difficulty=atoi(val);
    jstr(resp,"reward",val,sizeof(val));work->reward=atoi(val);
    jstr(resp,"timestamp",val,sizeof(val));work->timestamp=atoll(val);
    jstr(resp,"last_timestamp",val,sizeof(val));work->last_timestamp=atoll(val);
    const char* ta=strstr(resp,"\"transactions\":");
    if(ta){const char* br=strchr(ta,'[');if(br){br++;char items[512]={0};int ilen=0;
        while(*br&&*br!=']'){while(*br==' '||*br==','||*br=='\n'||*br=='\r')br++;
        if(*br=='"'){br++;while(*br&&*br!='"'&&ilen<(int)sizeof(items)-2)items[ilen++]=*br++;
        if(*br=='"')br++;while(*br==' ')br++;if(*br==',')items[ilen++]=';';}}
        items[ilen]='\0';strncpy(work->transactions,items,sizeof(work->transactions)-1);}}
    if(!work->transactions[0])strcpy(work->transactions,"empty-block");return 1;
}

typedef struct{int success;int block_index;int reward;char hash[128];char error[256];int http_status;}SubmitResult;

static void submit_block(const char* host,int port,const WorkTemplate* work,
    long long nonce,long long ts,const char* hash_hex,const char* address,SubmitResult* result){
    char tx_json[512],tx_copy[512];strncpy(tx_copy,work->transactions,sizeof(tx_copy)-1);tx_copy[511]=0;
    tx_json[0]='[';int jpos=1;char* saveptr=NULL;char* tok=strtok_r(tx_copy,";",&saveptr);int first=1;
    while(tok){if(!first)tx_json[jpos++]=',';tx_json[jpos++]='"';while(*tok&&jpos<(int)sizeof(tx_json)-4)tx_json[jpos++]=*tok++;tx_json[jpos++]='"';first=0;tok=strtok_r(NULL,";",&saveptr);}
    tx_json[jpos++]=']';tx_json[jpos]='\0';
    char body[2048];
    sprintf(body,"{\"block_index\":%d,\"nonce\":%lld,\"hash\":\"%s\",\"miner_address\":\"%s\",\"timestamp\":%lld,\"transactions\":%s}",
        work->block_index,(long long)nonce,hash_hex,address,(long long)ts,tx_json);
    static char resp[2048];memset(resp,0,sizeof(resp));int status=0;
    http_request("POST",host,port,"/mining/submit",body,resp,sizeof(resp),&status);
    result->http_status=status;result->success=0;result->error[0]='\0';result->hash[0]='\0';
    if(jbool(resp,"success")==1){result->success=1;char val[64];
        jstr(resp,"block_index",val,sizeof(val));result->block_index=atoi(val);
        jstr(resp,"reward",val,sizeof(val));result->reward=atoi(val);
        jstr(resp,"hash",result->hash,sizeof(result->hash));}
    else{jstr(resp,"error",result->error,sizeof(result->error));if(!result->error[0])strncpy(result->error,resp,sizeof(result->error)-1);}
}

static double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9;
}

static void msleep(int ms) { usleep(ms*1000); }

/* ── Main ────────────────────────────────────────────────────────── */
int main(int argc, char** argv) {
    char address[128]="", node_host[128]="51.91.122.48";
    int node_port=8080; char work_path[128]="/mining/work";
    int gpu_id=-1; /* -1 = auto */

    for(int i=1;i<argc;i++){
        if((strcmp(argv[i],"--address")==0||strcmp(argv[i],"-a")==0)&&i+1<argc) strncpy(address,argv[++i],sizeof(address)-1);
        else if((strcmp(argv[i],"--node")==0||strcmp(argv[i],"-n")==0)&&i+1<argc){
            i++;char* url=argv[i];
            if(strncmp(url,"https://",8)==0){url+=8;node_port=443;}
            else if(strncmp(url,"http://",7)==0){url+=7;node_port=80;}
            char* col=strrchr(url,':');char* sl=strchr(url,'/');
            if(col&&(!sl||col<sl)){int hl=(int)(col-url);strncpy(node_host,url,hl);node_host[hl]='\0';node_port=atoi(col+1);}
            else strncpy(node_host,url,sizeof(node_host)-1);
        }
        else if(strcmp(argv[i],"--gpu")==0&&i+1<argc){gpu_id=atoi(argv[++i]);}
        else if(strncmp(argv[i],"SCO",3)==0) strncpy(address,argv[i],sizeof(address)-1);
    }
    if(!address[0]){printf("SCO address: ");scanf("%127s",address);}

    printf("\n=== Scorbits GPU Miner v7 — RTX 30/40 Linux (vast.ai) ===\n");
    printf("Node: %s:%d | Address: %s\n\n",node_host,node_port,address);

    int dev_count=0; cudaGetDeviceCount(&dev_count);
    if(dev_count==0){printf("[ERROR] No CUDA GPU!\n");return 1;}
    if(gpu_id<0) gpu_id=0;
    if(gpu_id>=dev_count){printf("[ERROR] GPU %d not found (%d available)\n",gpu_id,dev_count);return 1;}
    cudaSetDevice(gpu_id);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,gpu_id);
    printf("[GPU %d] %s | %d SMs | CUDA %d.%d | %dMB\n",
        gpu_id,prop.name,prop.multiProcessorCount,prop.major,prop.minor,(int)(prop.totalGlobalMem/1048576));

    /* RTX 30/40 can handle massive grids */
    int tpb=256, bpg=prop.multiProcessorCount*64;
    if(bpg>32768) bpg=32768;
    long long batch=(long long)tpb*bpg;
    printf("[GPU %d] Batch: %lld hashes/launch (%d blocks x %d threads)\n\n",gpu_id,(unsigned long long)batch,bpg,tpb);

    /* Self-test */
    {char h[65];
    sha256_cpu("hello",h);printf("[Test] hello: %s\n",strcmp(h,"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")==0?"OK":"WRONG");
    sha256_cpu("abc",h);printf("[Test] abc: %s\n",strcmp(h,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")==0?"OK":"WRONG");
    sha256_cpu("25141776309785empty-block0000000d65acc094259d1c9f653b5e761cb606c780e92746c2088e2fc4e2085527790666SCO7e38c262a4a26f838a5eb2c9a7876efd",h);
    printf("[Test] mining: %s\n\n",strcmp(h,"27d37432f9539b33661ed923979d75a9b9568aed29b1c159fc976372a4460410")==0?"OK":"WRONG");}

    /* GPU buffers */
    uint32_t *d_midstate;char *d_tail,*d_suffix,*d_found_hash;
    long long *d_found_nonce,*d_found_ts_dev;
    cudaMalloc(&d_midstate,8*sizeof(uint32_t));cudaMalloc(&d_tail,2048);cudaMalloc(&d_suffix,256);
    cudaMalloc(&d_found_nonce,sizeof(long long));cudaMalloc(&d_found_ts_dev,sizeof(long long));cudaMalloc(&d_found_hash,65);

    WorkTemplate work; long long last_accepted_ts=0,total_blocks=0; double session_start=get_time();

    for(;;){
        printf("[Work] Fetching...\n");memset(&work,0,sizeof(work));
        if(!fetch_work(node_host,node_port,work_path,address,&work)){printf("[Work] Failed — 5s\n");msleep(5000);continue;}
        printf("[Work] #%d | diff=%d | reward=%d | lastTs=%lld\n",work.block_index,work.difficulty,work.reward,(long long)work.last_timestamp);
        if(work.last_timestamp>last_accepted_ts) last_accepted_ts=work.last_timestamp;
        /* Only track OUR accepted blocks for anti-spike — not others' */

        /* No pre-wait — keep GPU mining immediately!
         * We will wait only right before submitting if needed */

        long long ts=(long long)time(NULL);
        char prefix[2048];int plen=snprintf(prefix,sizeof(prefix),"%d%lld%s%s",work.block_index,(long long)ts,work.transactions,work.previous_hash);
        Midstate ms;compute_midstate(prefix,plen,&ms);
        printf("[Midstate] prefix=%d covered=%d tail=%d\n",plen,plen-ms.tail_len,ms.tail_len);

        int slen=(int)strlen(address),smem_size=ms.tail_len+slen;
        cudaMemcpy(d_midstate,ms.h,8*sizeof(uint32_t),cudaMemcpyHostToDevice);
        cudaMemcpy(d_tail,ms.tail,ms.tail_len,cudaMemcpyHostToDevice);
        cudaMemcpy(d_suffix,address,slen,cudaMemcpyHostToDevice);

        long long h_nonce=-1LL,h_found_ts=0;char h_hash[65]={0};
        cudaMemcpy(d_found_nonce,&h_nonce,sizeof(long long),cudaMemcpyHostToDevice);
        cudaMemcpy(d_found_ts_dev,&h_found_ts,sizeof(long long),cudaMemcpyHostToDevice);
        cudaMemset(d_found_hash,0,65);

        printf("[Miner] Mining #%d (diff=%d) on GPU %d...\n",work.block_index,work.difficulty,gpu_id);
        long long base=0,batch_hashes=0;static long long global_base=0;
        double t0=get_time(),tr=t0,poll=t0;int found=0;

        for(;;){
            long long new_ts=(long long)time(NULL);
            if(new_ts!=ts){
                ts=new_ts;plen=snprintf(prefix,sizeof(prefix),"%d%lld%s%s",work.block_index,ts,work.transactions,work.previous_hash);
                compute_midstate(prefix,plen,&ms);
                cudaMemcpy(d_midstate,ms.h,8*sizeof(uint32_t),cudaMemcpyHostToDevice);
                cudaMemcpy(d_tail,ms.tail,ms.tail_len,cudaMemcpyHostToDevice);
                smem_size=ms.tail_len+slen;
            }

            mine_kernel<<<bpg,tpb,smem_size>>>(d_midstate,d_tail,ms.tail_len,d_suffix,slen,ms.total_prefix_len,
                base,work.difficulty,d_found_nonce,d_found_ts_dev,ts,d_found_hash);
            cudaDeviceSynchronize();
            batch_hashes+=batch;base+=batch;global_base=base;

            cudaMemcpy(&h_nonce,d_found_nonce,sizeof(long long),cudaMemcpyDeviceToHost);
            if(h_nonce!=-1LL){
                cudaMemcpy(h_hash,d_found_hash,64,cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_found_ts,d_found_ts_dev,sizeof(long long),cudaMemcpyDeviceToHost);
                h_hash[64]='\0';found=1;break;
            }

            double now2=get_time();
            if(now2-tr>=3.0){
                double el=now2-t0,hr=batch_hashes/el;
                printf("[GPU %d] #%d | %.2f MH/s | %lld H | diff=%d\n",gpu_id,work.block_index,hr/1e6,(long long)batch_hashes,work.difficulty);
                tr=now2;
            }

            /* 3s poll — detect block AND difficulty changes */
            if(now2-poll>=3.0){
                poll=now2;WorkTemplate fresh;memset(&fresh,0,sizeof(fresh));
                if(fetch_work(node_host,node_port,work_path,address,&fresh)){
                    int new_block=fresh.block_index!=work.block_index;
                    int new_diff=fresh.difficulty!=work.difficulty;
                    if(new_block||new_diff){
                        if(new_block) printf("[Chain] #%d -> #%d\n",work.block_index,fresh.block_index);
                        if(new_diff) printf("[DiffChange] %d -> %d mid-block!\n",work.difficulty,fresh.difficulty);
                        work=fresh;
                        h_nonce=-1LL;cudaMemcpy(d_found_nonce,&h_nonce,sizeof(long long),cudaMemcpyHostToDevice);
                        cudaMemset(d_found_hash,0,65);
                        base=global_base;batch_hashes=0;t0=get_time();tr=t0;
                        ts=(long long)time(NULL);
                        plen=snprintf(prefix,sizeof(prefix),"%d%lld%s%s",work.block_index,ts,work.transactions,work.previous_hash);
                        compute_midstate(prefix,plen,&ms);
                        cudaMemcpy(d_midstate,ms.h,8*sizeof(uint32_t),cudaMemcpyHostToDevice);
                        cudaMemcpy(d_tail,ms.tail,ms.tail_len,cudaMemcpyHostToDevice);
                        smem_size=ms.tail_len+slen;
                    }
                }
            }
        }

        if(!found) continue;

        double el=get_time()-t0;
        printf("[Found!] #%d nonce=%lld ts=%lld %.1fs %.2f MH/s\n",work.block_index,(long long)h_nonce,(long long)h_found_ts,el,(double)batch_hashes/el/1e6);
        printf("[Hash] GPU: %s\n",h_hash);

        char verify_in[2048];snprintf(verify_in,sizeof(verify_in),"%d%lld%s%s%lld%s",work.block_index,(long long)h_found_ts,work.transactions,work.previous_hash,(long long)h_nonce,address);
        char cpu_hash[65];sha256_cpu(verify_in,cpu_hash);printf("[Hash] CPU: %s\n",cpu_hash);
        if(strcmp(h_hash,cpu_hash)!=0){printf("[ERROR] GPU/CPU mismatch!\n");
            h_nonce=-1LL;cudaMemcpy(d_found_nonce,&h_nonce,sizeof(long long),cudaMemcpyHostToDevice);cudaMemset(d_found_hash,0,65);continue;}
        printf("[Verify] OK!\n");

        /* Wait for anti-spike window using chain's last_timestamp */
        {
            long long chain_last = work.last_timestamp;
            long long need_ts = chain_last + 121;
            long long now_t = (long long)time(NULL);
            if(now_t < need_ts){
                long long wait = need_ts - now_t;
                printf("[AntiSpike] Wait %llds (chain lastTs=%lld)...\n", wait, chain_last);
                /* Keep GPU mining during wait — find a FRESH block for when window opens */
                long long wait_base = base;
                double wait_t0 = get_time();
                long long wait_hashes = 0;
                while((long long)time(NULL) < need_ts){
                    long long new_ts2=(long long)time(NULL);
                    if(new_ts2!=ts){
                        ts=new_ts2;
                        plen=snprintf(prefix,sizeof(prefix),"%d%lld%s%s",work.block_index,(long long)ts,work.transactions,work.previous_hash);
                        compute_midstate(prefix,plen,&ms);
                        cudaMemcpy(d_midstate,ms.h,8*sizeof(uint32_t),cudaMemcpyHostToDevice);
                        cudaMemcpy(d_tail,ms.tail,ms.tail_len,cudaMemcpyHostToDevice);
                        smem_size=ms.tail_len+slen;
                    }
                    /* Check for newer block from chain */
                    WorkTemplate fresh2;memset(&fresh2,0,sizeof(fresh2));
                    if(fetch_work(node_host,node_port,work_path,address,&fresh2)){
                        if(fresh2.block_index!=work.block_index){
                            printf("[AntiSpike] Chain moved to #%d — switching\n",fresh2.block_index);
                            work=fresh2; found=0; break;
                        }
                        if(fresh2.last_timestamp > chain_last){
                            chain_last=fresh2.last_timestamp;
                            need_ts=chain_last+121;
                            work.last_timestamp=chain_last;
                            printf("[AntiSpike] lastTs updated to %lld — new wait %llds\n",
                                chain_last,(long long)(need_ts-(long long)time(NULL)));
                        }
                    }
                    if(!found) break;
                    /* Reset found and keep mining for fresh block */
                    h_nonce=-1LL;
                    cudaMemcpy(d_found_nonce,&h_nonce,sizeof(long long),cudaMemcpyHostToDevice);
                    cudaMemset(d_found_hash,0,65);
                    mine_kernel<<<bpg,tpb,smem_size>>>(d_midstate,d_tail,ms.tail_len,d_suffix,slen,ms.total_prefix_len,
                        wait_base,work.difficulty,d_found_nonce,d_found_ts_dev,ts,d_found_hash);
                    cudaDeviceSynchronize();
                    wait_hashes+=batch; wait_base+=batch; base=wait_base; global_base=base;
                    cudaMemcpy(&h_nonce,d_found_nonce,sizeof(long long),cudaMemcpyDeviceToHost);
                    if(h_nonce!=-1LL){
                        cudaMemcpy(h_hash,d_found_hash,64,cudaMemcpyDeviceToHost);
                        cudaMemcpy(&h_found_ts,d_found_ts_dev,sizeof(long long),cudaMemcpyDeviceToHost);
                        h_hash[64]='\0';
                        /* Verify new hash */
                        char vi2[2048];snprintf(vi2,sizeof(vi2),"%d%lld%s%s%lld%s",work.block_index,(long long)h_found_ts,work.transactions,work.previous_hash,(long long)h_nonce,address);
                        char ch2[65];sha256_cpu(vi2,ch2);
                        if(strcmp(h_hash,ch2)==0){
                            printf("[AntiSpike] Fresh block found! nonce=%lld ts=%lld\n",(long long)h_nonce,(long long)h_found_ts);
                        }
                    }
                    double wr=get_time()-wait_t0;
                    if(wr>0) printf("[Waiting] %.2f MH/s | %llds left\n",
                        (double)wait_hashes/wr/1e6,(long long)(need_ts-(long long)time(NULL)));
                    msleep(100);
                }
            }
        }
        if(!found) continue;

        printf("[Submit] ts=%lld now=%lld\n",(long long)h_found_ts,(long long)time(NULL));
        double retry_start=get_time();int submitted_index=work.block_index;int submit_attempts=0;
        for(;;){
            if(get_time()-retry_start>90.0){printf("[Submit] Timeout\n");break;}
            if((int)(get_time()-retry_start)%10==0&&get_time()-retry_start>5.0){
                WorkTemplate check;memset(&check,0,sizeof(check));
                if(fetch_work(node_host,node_port,work_path,address,&check)&&check.block_index!=submitted_index){printf("[Stale] #%d\n",check.block_index);break;}
            }
            SubmitResult sr;memset(&sr,0,sizeof(sr));
            submit_block(node_host,node_port,&work,h_nonce,h_found_ts,h_hash,address,&sr);
            submit_attempts++;
            if(sr.success){total_blocks++;last_accepted_ts=(long long)time(NULL);
                printf("[Accepted] #%d +%d SCO Total:%lld\n",sr.block_index,sr.reward,(long long)total_blocks);
                printf("[Stats] %.0fs %lld blocks\n",get_time()-session_start,(long long)total_blocks);break;}
            else if(sr.http_status==409){printf("[Stale]\n");break;}
            else if(sr.http_status==429){printf("[RateLimit] 30s\n");msleep(30000);break;}
            else if(strstr(sr.error,"rapide")||strstr(sr.error,"anti-spike")||strstr(sr.error,"spike")){
                printf("[Rejected] %s — fetching fresh work\n",sr.error);
                /* Don't wait idle — break and re-mine with correct timestamp */
                break;
            }
            else{printf("[Rejected] HTTP=%d %s\n",sr.http_status,sr.error);break;}
        }
    }

    cudaFree(d_midstate);cudaFree(d_tail);cudaFree(d_suffix);
    cudaFree(d_found_nonce);cudaFree(d_found_hash);cudaFree(d_found_ts_dev);
    return 0;
}

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

static const char* src =
"__kernel void matmul_bias(__global const float* A,__global const float* Bb,__global const float* bias,__global float* C,const int M,const int K,const int NN){\n"
"  int m=get_global_id(0); int n=get_global_id(1);\n"
"  if(m<M&&n<NN){ float s=0; for(int k=0;k<K;k++) s+=A[m*K+k]*Bb[k*NN+n]; C[m*NN+n]=s+bias[n]; }\n"
"}\n";

static void die(const char* m){ fprintf(stderr,"ERR: %s\n",m); exit(1); }

/* model hyperparams */
static int VOCAB, D, NH, NL, BLOCK, FFN, DH;
static char** vocab;            /* id -> token string */
static int* inv;                /* token string -> id (linear) */
static float *ln1g,*ln1b,*ln2g,*ln2b,*lnfg,*lnfb;
static cl_mem gWq,gWk,gWv,gWo,gW1,gW2,gWlm;
static float *bq,*bk,*bv,*bo,*b1,*b2,*blm;
static float *tok_emb,*pos_emb;
static cl_context ctx; static cl_command_queue q; static cl_kernel kmb;
static cl_mem bufA, bufBias, bufC;
static float* smp_p; static int* smp_ord;

static float* xmalloc(size_t n){ float* p=malloc(n); if(!p) die("oom"); return p; }
static float* rd(FILE* f, size_t n){ float* p=xmalloc(n*4); if(fread(p,4,n,f)!=(size_t)n) die("read"); return p; }
static cl_mem rd_weight(FILE* f, int out, int in){
    float* w=rd(f,(size_t)out*in);
    float* t=xmalloc((size_t)in*out*4);
    for(int i=0;i<in;i++) for(int o=0;o<out;o++) t[i*out+o]=w[o*in+i];
    free(w);
    cl_int err; cl_mem b=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)in*out*4,t,&err);
    if(!b) die("buf"); free(t); return b;
}

static void gpu_mm(float* A, int M, int K, cl_mem B, int NN, float* bias, float* out){
    clEnqueueWriteBuffer(q,bufA,CL_TRUE,0,(size_t)M*K*4,A,0,NULL,NULL);
    clEnqueueWriteBuffer(q,bufBias,CL_TRUE,0,(size_t)NN*4,bias,0,NULL,NULL);
    clSetKernelArg(kmb,0,sizeof(cl_mem),&bufA);
    clSetKernelArg(kmb,1,sizeof(cl_mem),&B);
    clSetKernelArg(kmb,2,sizeof(cl_mem),&bufBias);
    clSetKernelArg(kmb,3,sizeof(cl_mem),&bufC);
    clSetKernelArg(kmb,4,sizeof(int),&M);
    clSetKernelArg(kmb,5,sizeof(int),&K);
    clSetKernelArg(kmb,6,sizeof(int),&NN);
    size_t g[2]={(size_t)M,(size_t)NN};
    clEnqueueNDRangeKernel(q,kmb,2,NULL,g,NULL,0,NULL,NULL);
    clFinish(q);
    clEnqueueReadBuffer(q,bufC,CL_TRUE,0,(size_t)M*NN*4,out,0,NULL,NULL);
}

static void ln(float* x, int T, const float* g, const float* b, float* y){
    for(int r=0;r<T;r++){
        float* xr=x+r*D; float* yr=y+r*D;
        float mu=0; for(int i=0;i<D;i++) mu+=xr[i]; mu/=D;
        float va=0; for(int i=0;i<D;i++){ float d=xr[i]-mu; va+=d*d; } va/=D;
        float inv=1.0f/sqrtf(va+1e-5f);
        for(int i=0;i<D;i++) yr[i]=(xr[i]-mu)*inv*g[i]+b[i];
    }
}
static inline float gelu(float x){ return 0.5f*x*(1.0f+tanhf(0.7978845608f*(x+0.044715f*x*x*x))); }
static void addto(float* x, float* y, int n){ for(int i=0;i<n;i++) x[i]+=y[i]; }

static void attention(float* Q, float* K, float* V, int T, float* attn_out){
    float scale=1.0f/sqrtf((float)DH);
    float* S=xmalloc((size_t)T*T*4);
    float* ctx=xmalloc((size_t)T*DH*4);
    for(int h=0;h<NH;h++){
        int off=h*DH;
        for(int i=0;i<T;i++){
            for(int j=0;j<T;j++){
                float s=0; for(int k=0;k<DH;k++) s+=Q[i*D+off+k]*K[j*D+off+k];
                if(j>i) s=-1e30f; else s*=scale;
                S[i*T+j]=s;
            }
            float mx=S[i*T+0]; for(int j=1;j<T;j++) if(S[i*T+j]>mx) mx=S[i*T+j];
            float sum=0; for(int j=0;j<T;j++){ float e=expf(S[i*T+j]-mx); S[i*T+j]=e; sum+=e; }
            for(int j=0;j<T;j++) S[i*T+j]/=sum;
            for(int k=0;k<DH;k++){ float s=0; for(int j=0;j<T;j++) s+=S[i*T+j]*V[j*D+off+k]; ctx[i*DH+k]=s; }
        }
        for(int i=0;i<T;i++) for(int k=0;k<DH;k++) attn_out[i*D+off+k]=ctx[i*DH+k];
    }
    free(S); free(ctx);
}

static void forward(int* ids, int T, float* logits){
    int n=T*FFN;
    float* x=xmalloc((size_t)n*4);
    float* h=xmalloc((size_t)n*4);
    float* Q=xmalloc((size_t)n*4);
    float* K=xmalloc((size_t)n*4);
    float* attnV=xmalloc((size_t)n*4);
    float* ao=xmalloc((size_t)n*4);
    float* tmp=xmalloc((size_t)n*4);
    for(int i=0;i<T;i++){
        int tid=ids[i]; int pid=i%BLOCK;
        for(int d=0;d<D;d++) x[i*D+d]=tok_emb[tid*D+d]+pos_emb[pid*D+d];
    }
    for(int L=0;L<NL;L++){
        ln(x,T,ln1g,ln1b,h);
        gpu_mm(h,T,D,gWq,D,bq,Q);
        gpu_mm(h,T,D,gWk,D,bk,K);
        gpu_mm(h,T,D,gWv,D,bv,attnV);
        attention(Q,K,attnV,T,ao);
        gpu_mm(ao,T,D,gWo,D,bo,tmp);
        addto(x,tmp,n);
        ln(x,T,ln2g,ln2b,h);
        gpu_mm(h,T,D,gW1,FFN,b1,tmp);
        for(int i=0;i<n;i++) tmp[i]=gelu(tmp[i]);
        gpu_mm(tmp,T,FFN,gW2,D,b2,h);
        addto(x,h,n);
    }
    ln(x,T,lnfg,lnfb,h);
    gpu_mm(h+(T-1)*D, 1, D, gWlm, VOCAB, blm, logits);
    free(x);free(h);free(Q);free(K);free(attnV);free(ao);free(tmp);
}

/* ---- word tokenizer (mirrors Python: lowercase, [a-z0-9]+ or single punct) ---- */
static int tokenize(const char* s, char toks[][32], int maxn){
    int n=0, bl=0; char buf[32];
    for(int i=0; s[i]; i++){
        unsigned char c=(unsigned char)s[i];
        if(c>='A'&&c<='Z') c=c+32;
        int alnum = (c>='a'&&c<='z')||(c>='0'&&c<='9');
        if(c==' '||c=='\t'||c=='\n'||c=='\r'){
            if(bl){ buf[bl]=0; if(n<maxn) strcpy(toks[n++],buf); bl=0; }
        } else if(alnum){
            if(bl<31) buf[bl++]=c;
        } else {
            if(bl){ buf[bl]=0; if(n<maxn) strcpy(toks[n++],buf); bl=0; }
            if(n<maxn){ buf[0]=(char)c; buf[1]=0; strcpy(toks[n++],buf); }
        }
    }
    if(bl){ buf[bl]=0; if(n<maxn) strcpy(toks[n++],buf); }
    return n;
}
static int enc_str(const char* s, int* ids, int maxn){
    char toks[4096][32];
    int nt=tokenize(s,toks,4096);
    int n=0;
    for(int i=0;i<nt;i++){
        for(int v=0;v<VOCAB;v++){ if(strcmp(vocab[v],toks[i])==0){ if(n<maxn) ids[n++]=v; goto nx; } }
        if(n<maxn) ids[n++]=0; /* <unk> */
        nx:;
    }
    return n;
}

static unsigned int rnd(void);
static int sample(float* lg, float temp, float topp){
    float mx=-1e30f; for(int i=0;i<VOCAB;i++){ smp_p[i]=lg[i]/temp; if(smp_p[i]>mx) mx=smp_p[i]; }
    float sum=0; for(int i=0;i<VOCAB;i++){ smp_p[i]=expf(smp_p[i]-mx); sum+=smp_p[i]; }
    for(int i=0;i<VOCAB;i++) smp_ord[i]=i;
    for(int a=0;a<VOCAB;a++) for(int b=a+1;b<VOCAB;b++) if(smp_p[smp_ord[b]]>smp_p[smp_ord[a]]){ int t=smp_ord[a]; smp_ord[a]=smp_ord[b]; smp_ord[b]=t; }
    float cum=0; int nk=0; int keep[16384];
    for(int k=0;k<VOCAB;k++){ int i=smp_ord[k]; cum+=smp_p[i]; keep[nk++]=i; if(cum>=topp*sum) break; }
    float tot=0; for(int k=0;k<nk;k++) tot+=smp_p[keep[k]];
    float r=(float)rnd()/4294967295.0f*tot; float acc=0;
    for(int k=0;k<nk;k++){ int i=keep[k]; acc+=smp_p[i]; if(acc>=r) return i; }
    return keep[0];
}

static unsigned int rngs=123456789;
static unsigned int rnd(){ rngs^=rngs<<13; rngs^=rngs>>17; rngs^=rngs<<5; return rngs; }

/* --- tiny safe arithmetic evaluator for math questions --- */
static double m_expr(const char** s);
static void gen_reply(const char* ctx, char* out, int outcap);
static void m_skip(const char** s){ while(**s==' '||**s=='\t') (*s)++; }
static double m_prim(const char** s){
    m_skip(s);
    if(**s=='('){ (*s)++; double v=m_expr(s); m_skip(s); if(**s==')') (*s)++; return v; }
    int neg=0; if(**s=='-'){ neg=1; (*s)++; }
    double v=0;
    if(isdigit((unsigned char)**s) || **s=='.'){ v=strtod(*s,(char**)s); }
    return neg? -v : v;
}
static double m_term(const char** s){
    double v=m_prim(s);
    for(;;){ m_skip(s);
        if(**s=='*'){ (*s)++; v*=m_prim(s); }
        else if(**s=='/'){ (*s)++; double d=m_prim(s); v=(d==0)?0:v/d; }
        else break;
    }
    return v;
}
static double m_expr(const char** s){
    double v=m_term(s);
    for(;;){ m_skip(s);
        if(**s=='+'){ (*s)++; v+=m_term(s); }
        else if(**s=='-'){ (*s)++; v-=m_term(s); }
        else break;
    }
    return v;
}
/* returns 1 and fills out if raw is a pure arithmetic expression, else 0 */
static int try_math(const char* raw, char* out, int cap){
    const char* p=raw; int has_op=0, len=0;
    while(*p){
        char c=*p++; len++;
        if((c>='0'&&c<='9')||c=='.'||c==' '||c=='\t'||c=='('||c==')') continue;
        if(c=='+'||c=='-'||c=='*'||c=='/'){ has_op=1; continue; }
        return 0;
    }
    if(!has_op || len==0) return 0;
    char* copy=strdup(raw); if(!copy) return 0;
    const char* ss=copy; double r=m_expr(&ss); m_skip(&ss);
    if(*ss!=0){ free(copy); return 0; }
    free(copy);
    if(fabs(r-(long long)r)<1e-9 && fabs(r)<1e15) snprintf(out,cap,"%lld",(long long)llround(r));
    else snprintf(out,cap,"%.4g",r);
    return 1;
}
static void compute_reply(const char* raw, char* out, int cap){
    if(try_math(raw,out,cap)) return;
    char ctx[8192]; snprintf(ctx,sizeof(ctx),"you : %s friend :",raw);
    gen_reply(ctx,out,cap);
}

static void gen_reply(const char* ctx, char* out, int outcap){
    int ids[9000];
    int bn=enc_str(ctx,ids,9000);
    if(bn>BLOCK){ memmove(ids,ids+bn-BLOCK,BLOCK*sizeof(int)); bn=BLOCK; }
    float* lg=xmalloc((size_t)VOCAB*4);
    int reply[2048]; int rl=0;
    for(int step=0; step<40; step++){
        forward(ids,bn,lg);
        int nxt=sample(lg, 0.7f, 0.9f);
        const char* w=vocab[nxt];
        if(strcmp(w,"\n")==0){ if(step==0) continue; else break; }
        reply[rl++]=nxt;
        ids[bn]=nxt; bn++; if(bn>BLOCK) bn=BLOCK;
    }
    int p=0;
    for(int i=0;i<rl;i++){
        const char* w=vocab[reply[i]];
        int wl=(int)strlen(w);
        if(wl==1 && !((w[0]>='a'&&w[0]<='z')||(w[0]>='0'&&w[0]<='9'))){ if(p+2<outcap){ out[p++]=w[0]; } }
        else { if(p+wl+1<outcap){ out[p++]=' '; strcpy(out+p,w); p+=wl; } }
    }
    if(p>0 && out[0]==' ') p--;
    out[p]=0;
    free(lg);
}

static void run_server(int port){
    int sv=socket(AF_INET,SOCK_STREAM,0); if(sv<0) die("socket");
    int opt=1; setsockopt(sv,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof opt);
    struct sockaddr_in a; a.sin_family=AF_INET; a.sin_port=htons((uint16_t)port); a.sin_addr.s_addr=INADDR_ANY;
    if(bind(sv,(struct sockaddr*)&a,sizeof a)) die("bind");
    if(listen(sv,8)) die("listen");
    fprintf(stderr,"server listening on :%d\n",port); fflush(stderr);
    while(1){
        int c=accept(sv,0,0); if(c<0) continue;
        char buf[8192]; int n=0; char ch;
        while(n<(int)sizeof(buf)-1 && read(c,&ch,1)==1 && ch!='\n' && ch!='\r') buf[n++]=ch;
        buf[n]=0;
        char out[8192]; compute_reply(buf,out,sizeof out);
        write(c,out,strlen(out)); write(c,"\n",1);
        close(c);
    }
}

int main(int argc, char** argv){
    const char* binname="tinyllm.bin";
    const char* server_port=NULL;
    const char* one_shot=NULL;
    int do_logits=0; const char* logits_s=NULL;
    for(int k=1;k<argc;k++){
        if(strcmp(argv[k],"--bin")==0 && k+1<argc){ binname=argv[k+1]; k++; }
        else if(strcmp(argv[k],"--server")==0 && k+1<argc){ server_port=argv[k+1]; k++; }
        else if(strcmp(argv[k],"--logits")==0 && k+1<argc){ do_logits=1; logits_s=argv[k+1]; k++; }
        else if(!one_shot && !server_port && !do_logits){ one_shot=argv[k]; }
    }
    FILE* f=fopen(binname,"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",binname); exit(1); }
    int magic,ver;
    if(fread(&magic,4,1,f)!=1||fread(&ver,4,1,f)!=1) die("hdr");
    if(magic!=0x4C4C4D31) die("magic");
    int fm;
    if(fread(&VOCAB,4,1,f)!=1||fread(&D,4,1,f)!=1||fread(&NH,4,1,f)!=1||fread(&NL,4,1,f)!=1||fread(&BLOCK,4,1,f)!=1||fread(&fm,4,1,f)!=1) die("hdr2");
    FFN=D*fm; DH=D/NH;
    vocab=malloc(sizeof(char*)*VOCAB);
    for(int i=0;i<VOCAB;i++){ int L=fgetc(f); if(L<0) die("vocab"); char* b=malloc(L+1); if(fread(b,1,L,f)!=(size_t)L) die("vocab"); b[L]=0; vocab[i]=b; }
    inv=NULL; /* linear search via vocab[] */
    smp_p=malloc(sizeof(float)*VOCAB); smp_ord=malloc(sizeof(int)*VOCAB);

    cl_platform_id p; cl_device_id dv; cl_int err;
    if(clGetPlatformIDs(1,&p,NULL)) die("plat");
    if(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&dv,NULL)) die("dev");
    ctx=clCreateContext(NULL,1,&dv,NULL,NULL,&err); if(!ctx) die("ctx");
    q=clCreateCommandQueue(ctx,dv,0,&err); if(!q) die("q");
    size_t fs=strlen(src);
    cl_program prog=clCreateProgramWithSource(ctx,1,&src,&fs,&err); if(!prog) die("prog");
    if(clBuildProgram(prog,1,&dv,NULL,NULL,NULL)){ char log[8192]; clGetProgramBuildInfo(prog,dv,CL_PROGRAM_BUILD_LOG,sizeof(log),log,NULL); fprintf(stderr,"BUILD:\n%s\n",log); exit(1);}
    kmb=clCreateKernel(prog,"matmul_bias",&err); if(!kmb) die("kern");
    bufA=clCreateBuffer(ctx,CL_MEM_READ_WRITE,(size_t)BLOCK*FFN*4,NULL,&err);
    bufBias=clCreateBuffer(ctx,CL_MEM_READ_WRITE,(size_t)VOCAB*4,NULL,&err);
    bufC=clCreateBuffer(ctx,CL_MEM_READ_WRITE,(size_t)BLOCK*VOCAB*4,NULL,&err);

    tok_emb=rd(f,(size_t)VOCAB*D);
    pos_emb=rd(f,(size_t)BLOCK*D);
    for(int L=0;L<NL;L++){
        ln1g=rd(f,D); ln1b=rd(f,D);
        gWq=rd_weight(f,D,D); bq=rd(f,D);
        gWk=rd_weight(f,D,D); bk=rd(f,D);
        gWv=rd_weight(f,D,D); bv=rd(f,D);
        gWo=rd_weight(f,D,D); bo=rd(f,D);
        ln2g=rd(f,D); ln2b=rd(f,D);
        gW1=rd_weight(f,FFN,D); b1=rd(f,FFN);
        gW2=rd_weight(f,D,FFN); b2=rd(f,D);
    }
    lnfg=rd(f,D); lnfb=rd(f,D);
    gWlm=rd_weight(f,VOCAB,D); blm=rd(f,VOCAB);
    fclose(f);

    if(do_logits){
        const char* s=logits_s; int ids[4096];
        int T=enc_str(s,ids,4096); if(T>BLOCK) T=BLOCK;
        float* lg=xmalloc((size_t)VOCAB*4); forward(ids,T,lg);
        printf("logits(last)[:8]="); for(int i=0;i<8&&i<VOCAB;i++) printf(" %.3f", lg[i]);
        printf("\n"); free(lg); return 0;
    }

    rngs=(unsigned int)time(NULL);
    if(server_port){ run_server(atoi(server_port)); return 0; }
    if(one_shot){
        char out[8192]; compute_reply(one_shot,out,sizeof out);
        printf("FRIEND: %s\n", out); fflush(stdout);
        return 0;
    }
    printf("hey! i'm your tiny gpu friend running on the fermi :)\n");
    char line[1024];
    while(1){
        printf("YOU: "); fflush(stdout);
        if(!fgets(line,sizeof(line),stdin)) break;
        int L=(int)strlen(line); while(L>0&&(line[L-1]=='\n'||line[L-1]=='\r')) line[--L]=0;
        if(L==0) continue;
        char out[8192]; compute_reply(line,out,sizeof out);
        printf("FRIEND: %s\n", out); fflush(stdout);
    }
    return 0;
}

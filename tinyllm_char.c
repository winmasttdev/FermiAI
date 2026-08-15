#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define BLK 4096
static const char* src =
"__kernel void matmul_bias(__global const float* A,__global const float* Bb,__global const float* bias,__global float* C,const int M,const int K,const int NN){\n"
"  int m=get_global_id(0); int n=get_global_id(1);\n"
"  if(m<M&&n<NN){ float s=0; for(int k=0;k<K;k++) s+=A[m*K+k]*Bb[k*NN+n]; C[m*NN+n]=s+bias[n]; }\n"
"}\n";

static void die(const char* m){ fprintf(stderr,"ERR: %s\n",m); exit(1); }

/* model hyperparams */
static int VOCAB, D, NH, NL, BLOCK, FFN, DH;
static char* vocab;            /* id -> char */
static int* inv;              /* char -> id (256 entries) */
/* weights (GPU buffers) */
static cl_mem gWq,gWk,gWv,gWo,gW1,gW2,gWlm;
static float *bq,*bk,*bv,*bo,*b1,*b2,*blm;
static float *tok_emb,*pos_emb;   /* CPU, for lookup */
static float *ln1g,*ln1b,*ln2g,*ln2b,*lnfg,*lnfb;

/* opencl */
static cl_context ctx; static cl_command_queue q; static cl_kernel kmb;
static cl_mem bufA, bufBias, bufC;

static float* xmalloc(size_t n){ float* p=malloc(n); if(!p) die("oom"); return p; }
static float* rd(FILE* f, size_t n){ float* p=xmalloc(n*4); if(fread(p,4,n,f)!=(size_t)n) die("read"); return p; }
/* read a (out x in) Linear weight and transpose to (in x out) for matmul */
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
static inline float gelu(float x){
    return 0.5f*x*(1.0f+tanhf(0.7978845608f*(x+0.044715f*x*x*x)));
}
static void addto(float* x, float* y, int n){ for(int i=0;i<n;i++) x[i]+=y[i]; }

/* attention on CPU: Q,K,V are T x D; writes attn_out T x D */
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
            /* softmax row i */
            float mx=S[i*T+0]; for(int j=1;j<T;j++) if(S[i*T+j]>mx) mx=S[i*T+j];
            float sum=0; for(int j=0;j<T;j++){ float e=expf(S[i*T+j]-mx); S[i*T+j]=e; sum+=e; }
            for(int j=0;j<T;j++) S[i*T+j]/=sum;
            /* ctx[i,:]=sum_j A[i,j]*V[j, head] */
            for(int k=0;k<DH;k++){ float s=0; for(int j=0;j<T;j++) s+=S[i*T+j]*V[j*D+off+k]; ctx[i*DH+k]=s; }
        }
        /* scatter into attn_out columns [off, off+DH) */
        for(int i=0;i<T;i++) for(int k=0;k<DH;k++) attn_out[i*D+off+k]=ctx[i*DH+k];
    }
    free(S); free(ctx);
}

/* forward over ids[0..T-1]; writes logits T x V */
static void forward(int* ids, int T, float* logits){
    int n=T*FFN;
    float* x=xmalloc((size_t)n*4);
    float* h=xmalloc((size_t)n*4);
    float* Q=xmalloc((size_t)n*4);
    float* K=xmalloc((size_t)n*4);
    float* attnV=xmalloc((size_t)n*4);
    float* ao=xmalloc((size_t)n*4);
    float* tmp=xmalloc((size_t)n*4);
    static int dbg=0;
    /* embedding */
    for(int i=0;i<T;i++){
        int tid=ids[i]; int pid=i%BLOCK;
        for(int d=0;d<D;d++) x[i*D+d]=tok_emb[tid*D+d]+pos_emb[pid*D+d];
    }
    for(int L=0;L<NL;L++){
        ln(x,T,ln1g,ln1b,h);
        gpu_mm(h,T,D,gWq,D,bq,Q);
        gpu_mm(h,T,D,gWk,D,bk,K);
        gpu_mm(h,T,D,gWv,D,bv,attnV);
        if(!dbg){ dbg=1; printf("DBG x0=%.4f h0=%.4f Q0=%.4f K0=%.4f V0=%.4f bq0=%.4f\n", x[0], h[0], Q[0], K[0], attnV[0], bq[0]); fflush(stdout); }
        attention(Q,K,attnV,T,ao);
        gpu_mm(ao,T,D,gWo,D,bo,tmp);
        addto(x,tmp,n);
        ln(x,T,ln2g,ln2b,h);
        gpu_mm(h,T,D,gW1,FFN,b1,tmp);
        for(int i=0;i<n;i++) tmp[i]=gelu(tmp[i]);
        gpu_mm(tmp,T,FFN,gW2,D,b2,h);
        addto(x,h,n);
        if(dbg){ printf("DBG ao0=%.4f tmp0=%.4f x0=%.4f\n", ao[0], tmp[0], x[0]); fflush(stdout); }
    }
    ln(x,T,lnfg,lnfb,h);
    gpu_mm(h,T,D,gWlm,VOCAB,blm,logits);
    if(dbg){ printf("DBG lnf0=%.4f blm0=%.4f logits_last[0..3]=%.4f,%.4f,%.4f,%.4f\n", h[0], blm[0], logits[(T-1)*VOCAB+0], logits[(T-1)*VOCAB+1], logits[(T-1)*VOCAB+2], logits[(T-1)*VOCAB+3]); fflush(stdout); }
    free(x);free(h);free(Q);free(K);free(attnV);free(ao);free(tmp);
}

/* ---------- sampling ---------- */
static unsigned int rngs=123456789;
static unsigned int rnd(){ rngs^=rngs<<13; rngs^=rngs>>17; rngs^=rngs<<5; return rngs; }
static int sample(float* lg, float temp, float topp){
    float p[BLK]; float mx=-1e30f;
    for(int i=0;i<VOCAB;i++){ p[i]=lg[i]/temp; if(p[i]>mx) mx=p[i]; }
    float sum=0;
    for(int i=0;i<VOCAB;i++){ p[i]=expf(p[i]-mx); sum+=p[i]; }
    int ord[BLK]; for(int i=0;i<VOCAB;i++) ord[i]=i;
    for(int a=0;a<VOCAB;a++) for(int b=a+1;b<VOCAB;b++) if(p[ord[b]]>p[ord[a]]){ int t=ord[a]; ord[a]=ord[b]; ord[b]=t; }
    float cum=0; int nk=0; int keep[BLK];
    for(int k=0;k<VOCAB;k++){ int i=ord[k]; cum+=p[i]; keep[nk++]=i; if(cum>=topp*sum) break; }
    float tot=0; for(int k=0;k<nk;k++) tot+=p[keep[k]];
    float r=(float)rnd()/4294967295.0f*tot; float acc=0;
    for(int k=0;k<nk;k++){ int i=keep[k]; acc+=p[i]; if(acc>=r) return i; }
    return keep[0];
}

static int enc(char c){ if((unsigned char)c<256 && inv[(unsigned char)c]>=0) return inv[(unsigned char)c]; return 0; }

int main(int argc, char** argv){
    /* load model */
    FILE* f=fopen("tinyllm.bin","rb"); if(!f) die("tinyllm.bin");
    int magic,ver;
    if(fread(&magic,4,1,f)!=1||fread(&ver,4,1,f)!=1) die("hdr");
    if(magic!=0x4C4C4D31) die("magic");
    int fm;
    if(fread(&VOCAB,4,1,f)!=1||fread(&D,4,1,f)!=1||fread(&NH,4,1,f)!=1||fread(&NL,4,1,f)!=1||fread(&BLOCK,4,1,f)!=1||fread(&fm,4,1,f)!=1) die("hdr2");
    FFN = D*fm;
    DH=D/NH;
    vocab=malloc(VOCAB+1); if(fread(vocab,1,VOCAB,f)!=(size_t)VOCAB) die("vocab");
    inv=malloc(256*sizeof(int)); for(int i=0;i<256;i++) inv[i]=-1;
    for(int i=0;i<VOCAB;i++) inv[(unsigned char)vocab[i]]=i;

    /* opencl */
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
    bufBias=clCreateBuffer(ctx,CL_MEM_READ_WRITE,(size_t)BLK*4,NULL,&err);
    bufC=clCreateBuffer(ctx,CL_MEM_READ_WRITE,(size_t)BLOCK*BLK*4,NULL,&err);

    /* weights */
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

    if(argc>1 && strcmp(argv[1],"--logits")==0){
        const char* s=argv[2]; int T=strlen(s); if(T>BLOCK) T=BLOCK;
        int* ids=malloc(T*sizeof(int)); for(int i=0;i<T;i++) ids[i]=enc(s[i]);
        float* lg=xmalloc((size_t)T*VOCAB*4); forward(ids,T,lg);
        printf("logits(last)[:8]="); for(int i=0;i<8&&i<VOCAB;i++) printf(" %.3f", lg[(T-1)*VOCAB+i]);
        printf("\n"); free(ids); free(lg); return 0;
    }

    /* chat */
    rngs=(unsigned int)time(NULL);
    printf("hey! i'm your tiny gpu friend running on the fermi :)\n");
    char hist[4096]; hist[0]=0;
    char line[1024];
    if(argc>1){ snprintf(line,sizeof(line),"%s",argv[1]); }
    while(1){
        if(argc>1){ /* one-shot */ }
        else { printf("YOU: "); fflush(stdout); if(!fgets(line,sizeof(line),stdin)) break; }
        int L=strlen(line); while(L>0&&(line[L-1]=='\n'||line[L-1]=='\r')) line[--L]=0;
        if(L==0){ if(argc>1) break; continue; }
        /* build context */
        char ctxbuf[8192];
        snprintf(ctxbuf,sizeof(ctxbuf),"%sYou: %s\nFriend: ",hist,line);
        int T=strlen(ctxbuf); int s=0; if(T>BLOCK){ s=T-BLOCK; T=BLOCK; }
        int* ids=malloc((T+256)*sizeof(int));
        for(int i=0;i<T;i++) ids[i]=enc(ctxbuf[s+i]);
        /* generate */
        int maxn=160; int gen=0; char reply[1024]; int rl=0;
        float* lg=xmalloc((size_t)(BLOCK+1)*VOCAB*4);
        for(int step=0; step<maxn; step++){
            forward(ids,T,lg);
            int nxt=sample(lg+(size_t)(T-1)*VOCAB, 0.9f, 0.9f);
            char c=vocab[nxt];
            if(c=='\n') break;
            if(rl<(int)sizeof(reply)-1) reply[rl++]=c;
            ids[T]=nxt; T++; if(T>BLOCK) T=BLOCK;
            gen++;
        }
        reply[rl]=0;
        printf("FRIEND: %s\n", reply); fflush(stdout);
        /* update history */
        int hl=strlen(hist);
        snprintf(hist+hl, sizeof(hist)-hl, "You: %s\nFriend: %s\n", line, reply);
        if(strlen(hist)>3000){ memmove(hist, hist+strlen(hist)-2000, 2001); hist[2000]=0; }
        free(ids); free(lg);
        if(argc>1) break;
    }
    return 0;
}

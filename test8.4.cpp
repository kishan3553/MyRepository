// complete for smaller inputs (sampleInput1.txt)
using namespace std;
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <vector>
#include <deque>
#include <algorithm>

#define DELTA 0.8
#define ALPHA 0.8
#define VERTEX 36 
#define R 2.0


//class which holds diversity of a node
class DiverseNode
{
public:
    int v;
    float diversity;
    DiverseNode(int x, float y)
    {
        v=x;
        diversity=y;
    }
    void printDiversity()
    {
        printf("%d %f\n",v,diversity);
    }
    void printToFile()
    {
        ofstream  outfile; 
        outfile.open("output/output.txt",ios::out | ios::app);
        outfile<<v<<"\t"<<diversity<<"\n";
        outfile.close();
    }
};

// Structure to represent a min heap node
struct MinHeapNode
{
    int v;
    float dist;
};

// Structure to represent a min heap
struct MinHeap
{
    int size;    
    int capacity; 
    int *pos;    
    struct MinHeapNode **array;
};

// Structure to represent an adjacency list node
struct Node
{
    int v;
    float e;
    float s_val;
    struct Node* next;
};
 
struct List
{
    float c_val;
    struct Node *head; 
};
 
struct Graph
{
    int V;
    struct List* array;
};


Node* newNode(int dest,float weight)
{
    Node* newNode=(Node*)malloc(sizeof(Node));
    if(newNode==NULL)
    {
        cout<<"memory error"<<endl;
        exit(0);
    }
    newNode->v=dest;
    newNode->s_val=-1.0f;
    newNode->e=weight;
    newNode->next=NULL;
    return newNode;
}
 
//function to create graph.
Graph* createGraph(int V)
{
    Graph* graph=(Graph*)malloc(sizeof(Graph));
    if(graph==NULL)
    {
        cout<<"memory error"<<endl;
        exit(0);
    }
    graph->V=V;
    graph->array=(List*)malloc(V*sizeof(List));
    if(graph->array==NULL)
    {
        cout<<"memory error"<<endl;
        exit(0);
    }
    int i;
    for(i=0;i<V;++i)
        graph->array[i].head=NULL;
    return graph;
}
 
// Function to add edge to an undirected graph.
void addEdge(Graph* graph,int src,int dest,float weight)
{
    Node* newnode=newNode(dest,weight);
    newnode->next=graph->array[src].head;
    graph->array[src].head=newnode;

    newnode=newNode(src,weight);
    newnode->next=graph->array[dest].head;
    graph->array[dest].head=newnode;
}

int binarySearch(int arr[], int l, int r, int x)
{
   if (r >= l)
   {
        int mid = l + (r - l)/2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid-1, x);
        return binarySearch(arr, mid+1, r, x);
   }
   return -1;
}

int returnIndex(int *index,int u)
{
    return binarySearch(index,0,VERTEX-1,u);
}

void addAllEdges(struct Graph* graph,int *index)
{
    ifstream  infile;  
    int i,j,u,v;
    infile.open("sampleInput1.txt",ios::in);
    infile.seekg(0,ios::beg);
    while (!infile.eof()) 
    {
        infile >> i >> j;
        u=returnIndex(index,i);
        v=returnIndex(index,j);
        if(u==-1||v==-1)
        {
            printf("\nError in function addALLEdge() in retrieveing index\n");
            exit(0);
        }
        addEdge(graph,u,v,1);
    }
    infile.close();
}

 
// Function to print the adjacenncy list representation of graph.
void printGraph(Graph* graph)
{
    int v;
    for(v=0;v<graph->V;++v)
    {
        Node* p=graph->array[v].head;
        printf("\nv=%d C=%g head ",v,graph->array[v].c_val);
        while(p)
        {
            printf("-> %d e=%g s=%g",p->v,p->e,p->s_val);
            p=p->next;
        }
        printf("\n");
    }
}

void indexing(int *index)
{
    ifstream  infile;
    int i,maxElement=0,j=1;
    infile.open("sampleInput1.txt",ios::in);
    infile.seekg(0,ios::beg); //115
    while (!infile.eof()) 
    {
        infile >> i;
        if(maxElement<i)
            maxElement=i;
    }
    int tempIndex[maxElement+1]={0};
    infile.close();
    infile.open("sampleInput1.txt",ios::in);
    infile.seekg(0,ios::beg); //115
    while (!infile.eof()) 
    {
        infile >> i ;
        tempIndex[i]=i;
    }
    sort(tempIndex,tempIndex+(maxElement+1));
    for(i=0; i<=maxElement; i++)
    {
        if(tempIndex[i])
        {
            index[j]=tempIndex[i];
            j++;
        }
    }
    infile.close();
}


// A utility function to create a new Min Heap Node
struct MinHeapNode* newMinHeapNode(int v, float dist)
{
    struct MinHeapNode* minHeapNode =
        (struct MinHeapNode*) malloc(sizeof(struct MinHeapNode));
    if(minHeapNode==NULL)
    {
        printf("Memory error\n");
        exit(0);
    }
    minHeapNode->v = v;
    minHeapNode->dist = dist;
    return minHeapNode;
}

// A utility function to create a Min Heap
struct MinHeap* createMinHeap(int capacity)
{
    struct MinHeap* minHeap =
        (struct MinHeap*) malloc(sizeof(struct MinHeap));
    minHeap->pos = (int *)malloc(capacity * sizeof(int));
    minHeap->size = 0;
    minHeap->capacity = capacity;
    minHeap->array =
        (struct MinHeapNode**) malloc(capacity * sizeof(struct MinHeapNode*));
    return minHeap;
}

// A utility function to swap two nodes of min heap. Needed for min heapify
void swapMinHeapNode(struct MinHeapNode** a, struct MinHeapNode** b)
{
    struct MinHeapNode* t = *a;
    *a = *b;
    *b = t;
}

// A standard function to heapify at given idx
// This function also updates position of nodes when they are swapped.
// Position is needed for decreaseKey()
void minHeapify(struct MinHeap* minHeap, int idx)
{
    int smallest, left, right;
    smallest = idx;
    left = 2 * idx + 1;
    right = 2 * idx + 2;
    if (left < minHeap->size &&
        minHeap->array[left]->dist < minHeap->array[smallest]->dist )
    smallest = left;
    if (right < minHeap->size &&
        minHeap->array[right]->dist < minHeap->array[smallest]->dist )
    smallest = right;
    if (smallest != idx)
    {
        MinHeapNode *smallestNode = minHeap->array[smallest];
        MinHeapNode *idxNode = minHeap->array[idx];
        minHeap->pos[smallestNode->v] = idx;
        minHeap->pos[idxNode->v] = smallest;
        swapMinHeapNode(&minHeap->array[smallest], &minHeap->array[idx]);
        minHeapify(minHeap, smallest);
    }
}

// A utility function to check if the given minHeap is ampty or not
int isEmpty(struct MinHeap* minHeap)
{
    return minHeap->size == 0;
}

// Standard function to extract minimum node from heap
struct MinHeapNode* extractMin(struct MinHeap* minHeap)
{
    if (isEmpty(minHeap))
        return NULL;
    struct MinHeapNode* root = minHeap->array[0];
    struct MinHeapNode* lastNode = minHeap->array[minHeap->size - 1];
    minHeap->array[0] = lastNode;
    minHeap->pos[root->v] = minHeap->size-1;
    minHeap->pos[lastNode->v] = 0;
    --minHeap->size;
    minHeapify(minHeap, 0);
    return root;
}

// Function to decreasy dist value of a given vertex v. This function
// uses pos[] of min heap to get the current index of node in min heap
void decreaseKey(struct MinHeap* minHeap, int v, float dist)
{
    int i = minHeap->pos[v];
    minHeap->array[i]->dist = dist;
    while (i && minHeap->array[i]->dist < minHeap->array[(i - 1) / 2]->dist)
    {
        minHeap->pos[minHeap->array[i]->v] = (i-1)/2;
        minHeap->pos[minHeap->array[(i-1)/2]->v] = i;
        swapMinHeapNode(&minHeap->array[i], &minHeap->array[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

// A utility function to check if a given vertex
// 'v' is in min heap or not
bool isInMinHeap(struct MinHeap *minHeap, int v)
{
if (minHeap->pos[v] < minHeap->size)
    return true;
return false;
}


// The main function that calulates distances of shortest paths from src to all vertices.
void dijkstra(struct Graph* graph, int src, float *dist)
{
    int V = graph->V;
    int flag=0;
    struct MinHeap* minHeap = createMinHeap(V);
    for (int v = 0; v < V; ++v)
    {
        dist[v] = INT_MAX;
        minHeap->array[v] = newMinHeapNode(v, dist[v]);
        minHeap->pos[v] = v;
    }
    dist[src] = 0;
    decreaseKey(minHeap, src, dist[src]);
    minHeap->size = V;
    while (!isEmpty(minHeap))
    {
        struct MinHeapNode* minHeapNode = extractMin(minHeap);
        int u = minHeapNode->v;
        struct Node* currentNode = graph->array[u].head;
        while (currentNode != NULL)
        {
            /*if((dist[u] + currentNode->e)>R)
            {
                //printf(" dijkstra%d %g %g ",u,dist[u],currentNode->e);
                flag=1;
                break;
            }*/
            int v = currentNode->v;
            if (isInMinHeap(minHeap, v) && dist[u] != INT_MAX && 
                                        currentNode->e + dist[u] < dist[v])
            {
                dist[v] = dist[u] + currentNode->e;

                decreaseKey(minHeap, v, dist[v]);
            }
            currentNode = currentNode->next;
        }
        /*if(flag==1)
            break;*/
    }
}

//Function to find R-neighbour of all vertices of a graph.
Graph* R_neighbour(Graph* G)
{
    Graph* Gr=createGraph(G->V);
    float dist[G->V];
    for(int i=0;i<Gr->V;i++)
    {       
        dijkstra(G,i,dist);
        Node **currentNode=&Gr->array[i].head;
        for(int j=0;j<G->V;j++)
        {
            if(dist[j]<=R&&i!=j)
            {
                *currentNode=newNode(j,dist[j]);
                currentNode=&(*currentNode)->next;
            }
        }
    }
    return Gr;
}

float similarityPair(Graph* Gr, int u, int v)
{
    float l=0;
    Node* node=Gr->array[u].head;
    while(node)
    {
        if(node->v==v)
        {
            l=node->e;
            return pow(DELTA,l-1);
        }
        node=node->next;
    }
    return 0;
}

float Sup(Graph* G, Graph* Gr, int u, int v)
{
    float sum = 0,u_e,v_e;
    float dist_u[G->V];
    float dist_v[G->V];
    vector <int> vec;
    vector <int> :: iterator i;
    Node* current_u=G->array[u].head;
    Node* current_v=G->array[v].head;
    //#################################
    while(current_u)
    {
        if(current_u->v==v)
        {
            v_e = current_u->e;
            current_u->e=INT_MAX;
            break;
        }
        current_u = current_u->next;
    }
    dijkstra(G,u,dist_u);
    dist_u[u] = INT_MAX;
    if(current_u)
        current_u->e=v_e;
    //#################################
    while(current_v)
    {
        if(current_v->v==u)
        {
            u_e = current_v->e;
            current_v->e=INT_MAX;
            break;
        }
        current_v = current_v->next;
    }
    dijkstra(G,v,dist_v);
    dist_v[v] = INT_MAX;
    if(current_v)
        current_v->e=u_e;
    //###################################
    for(int i=0; i<Gr->V; i++)
    {
        if(dist_u[i]<=R && dist_v[i]<=R)
        {
            vec.push_back(i);
        }
    }
    for(i=vec.begin(); i!=vec.end(); ++i)
    {
        sum=sum + similarityPair(Gr,v,*i) * similarityPair(Gr,u,*i);
    }
    return sum;
}

void computeS_val(Graph* G, Graph* Gr)
{
    Node* node = NULL, *node1 = NULL;
    for(int i=0;i<Gr->V;i++)
    {
        node=Gr->array[i].head;
        while(node)
        {
            if(node->s_val==-1.0f)
            {
                node->s_val=Sup(G,Gr,i,node->v);
                int j=node->v;
                node1=Gr->array[j].head;
                while(node1)
                {
                    if(node1->v==i)
                    {
                        node1->s_val=node->s_val;
                        break;
                    }
                    node1=node1->next;
                }
            }
            node=node->next;
        }
    }
}

float Cup(Graph* Gr, int v)
{
    float sum=0;
    Node* node=Gr->array[v].head;
    while(node)
    {
        sum=sum + similarityPair(Gr,node->v,v);
        node=node->next;
    }
    return sum;
}

void computeC_val(Graph* Gr)
{
    for(int i=0; i<Gr->V; i++)
    {
        Gr->array[i].c_val=Cup(Gr,i);
    }
}

void createSortedQueue(Graph* Gr, deque <int>& Q)
{
    deque <float> Q1;
    for(int i=0; i<Gr->V; i++)
    {
        Q.push_back(i);
        Q1.push_back(Gr->array[i].c_val);
    }
    int k,j;
    for(int i=0; i<Gr->V-1; i++)
    {
        k=i;
        for(j=i+1; j<Gr->V; j++)
        {
            if(Q1[k]<Q1[j])
                k=j;
        }
        if(k!=j)
        {
            swap(Q[k],Q[i]);
            swap(Q1[k],Q1[i]);
        }
    }
}

int minimumDiversity(vector <DiverseNode>& T)
{
    vector <DiverseNode> :: iterator itr=T.begin();
    int index;
    float m=(*itr).diversity;
    for(int i=0; itr!=T.end(); ++itr)
    {
        if((*itr).diversity<m)
        {
            m=(*itr).diversity;
            index=i;
        }
        ++i;
    }
    return index;
}


float returnS_val(Graph *Gr,int u,int v)
{
    Node* node= Gr->array[u].head;
    while(node)
    {
        if(node->v==v)
            return node->s_val;
        node=node->next;
    }
    printf("Error\n");
    exit(0);
}

float upper(Graph* Gr, int u, int v)
{
    return 1-(ALPHA*(returnS_val(Gr,u,v)/Gr->array[u].c_val));
}

float numF(Graph* G,Graph *Gr,int u,int v)
{
    float sum = 0,s,v_e;
    float dist_u[G->V];
    Node* current_u=G->array[u].head;
    //#################################
    while(current_u)
    {
        if(current_u->v==v)
        {
            v_e = current_u->e;
            current_u->e=INT_MAX;
            break;
        }
        current_u = current_u->next;
    }
    dijkstra(G,u,dist_u);
    if(current_u)
        current_u->e=v_e;
    //##################################
    for(int i=0; i<Gr->V; i++)
    {
        if(dist_u[i]<=R)
        {
            s=similarityPair(Gr,u,i);
            sum=sum+s;
        }
    }
    return sum;
}


float F(Graph* G, Graph* Gr, int u, int v)
{
    float num=numF(G,Gr,u,v);
    if(!num)
        return 1.0;
    else
        return 1 - (ALPHA * (returnS_val(Gr,u,v)/num));
}


//Top-K diversity ranking algorithm.
vector <DiverseNode> topKDiversity(Graph*G, Graph* Gr, int K)
{
    deque <int> Q;
    deque <int> :: iterator itr; 
    createSortedQueue(Gr,Q);
    double lBound = 0, UP=0, D=0,f;
    vector <DiverseNode> T;
    for(itr=Q.begin(); itr!=Q.end(); ++itr)
    {
        UP=0; D=0;
        if(Gr->array[*itr].c_val<lBound)
            return T;
        Node* currentNode=Gr->array[*itr].head;
        while(currentNode)
        {
            UP=UP + min(1.0f,upper(Gr,currentNode->v,*itr));
            currentNode=currentNode->next;
        }
        if(UP<lBound)
            break;
        currentNode=Gr->array[*itr].head;
        //printf("%d  ",*itr);
        while(currentNode)
        {
            f=F(G,Gr,currentNode->v,*itr);
            //printf("F %lf",f);
            D=D + f;
            currentNode=currentNode->next;
        }
        //printf("\n");
        if(D>lBound)
            T.push_back(DiverseNode(*itr,D));
        if(T.size()>K)
        {            T.erase(T.begin()+minimumDiversity(T));
            lBound=T[minimumDiversity(T)].diversity;
        }
    }
    return T;
}

void output(vector <DiverseNode> T)
{
    for(vector <DiverseNode> :: iterator itr=T.begin(); itr!=T.end(); ++itr)
    {
        (*itr).printToFile();
    }
}


//this function sorts K diverse node passed as a vector list.
void sortQ_Diversity(vector <DiverseNode>& T)
{
    vector <DiverseNode> :: iterator itr_i;
    vector <DiverseNode> :: iterator itr_j;
    vector <DiverseNode> :: iterator itr_k;
    vector <DiverseNode> :: iterator itr_temp;
    for(itr_i=T.begin(); itr_i!=T.end(); ++itr_i)
    {
        itr_k=itr_i;
        itr_temp=itr_i;
        for(itr_j=++itr_temp; itr_j<T.end(); itr_j++)
        {
            if((*itr_k).diversity<(*itr_j).diversity)
                itr_k=itr_j;
        }
        if(itr_k!=itr_j)
        {
            swap(*itr_k,*itr_i);
        }
    }
}

int main()
{
    int K = 36;
    int indexMainGraph[VERTEX]={0};
    indexing(indexMainGraph);
    Graph* graph = createGraph(VERTEX);
    addAllEdges(graph,indexMainGraph);
    Graph* gr = createGraph(VERTEX);
    gr=R_neighbour(graph);
    computeS_val(graph, gr);
    computeC_val(gr);
    //printGraph(gr);
    vector <DiverseNode> T=topKDiversity(graph,gr,K);
    sortQ_Diversity(T);
    vector <DiverseNode> :: iterator itr;
    //printf("\n");
    for(itr=T.begin(); itr!=T.end(); ++itr)
    {
        (*itr).printDiversity();
    }
    //output(T);
    return 0;
}
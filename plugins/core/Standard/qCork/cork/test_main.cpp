#include "file_formats/files.h"

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <sstream>
using std::stringstream;
using std::string;

using std::ostream;

#include "cork.h"


void file2corktrimesh(
    const Files::FileMesh &in, CorkTriMesh *out
) {
    out->n_vertices = in.vertices.size();
    out->n_triangles = in.triangles.size();
    
    out->triangles = new uint[(out->n_triangles) * 3];
    out->vertices  = new float[(out->n_vertices) * 3];
    
    for(uint i=0; i<out->n_triangles; i++) {
        (out->triangles)[3*i+0] = in.triangles[i].a;
        (out->triangles)[3*i+1] = in.triangles[i].b;
        (out->triangles)[3*i+2] = in.triangles[i].c;
    }
    
    for(uint i=0; i<out->n_vertices; i++) {
        (out->vertices)[3*i+0] = static_cast<float>(in.vertices[i].pos.x);
        (out->vertices)[3*i+1] = static_cast<float>(in.vertices[i].pos.y);
        (out->vertices)[3*i+2] = static_cast<float>(in.vertices[i].pos.z);
    }
}

void corktrimesh2file(
    CorkTriMesh in, Files::FileMesh &out
) {
    out.vertices.resize(in.n_vertices);
    out.triangles.resize(in.n_triangles);
    
    for(uint i=0; i<in.n_triangles; i++) {
        out.triangles[i].a = in.triangles[3*i+0];
        out.triangles[i].b = in.triangles[3*i+1];
        out.triangles[i].c = in.triangles[3*i+2];
    }
    
    for(uint i=0; i<in.n_vertices; i++) {
        out.vertices[i].pos.x = in.vertices[3*i+0];
        out.vertices[i].pos.y = in.vertices[3*i+1];
        out.vertices[i].pos.z = in.vertices[3*i+2];
    }
}

void loadMesh(string filename, CorkTriMesh *out)
{
    Files::FileMesh filemesh;
    
    if(Files::readTriMesh(filename, &filemesh) > 0) {
        cerr << "Unable to load in " << filename << endl;
        exit(1);
    }
    
    file2corktrimesh(filemesh, out);
}
void saveMesh(string filename, CorkTriMesh in)
{
    Files::FileMesh filemesh;
    
    corktrimesh2file(in, filemesh);
    
    if(Files::writeTriMesh(filename, &filemesh) > 0) {
        cerr << "Unable to write to " << filename << endl;
        exit(1);
    }
}



//std::function< void(
//    std::vector<string>::iterator &,
//    const std::vector<string>::iterator &
//) >
//genericBinaryOp(
//    std::function< void(CorkTriMesh in0, CorkTriMesh in1, CorkTriMesh *out) >
//        binop
//) {
//    return [binop]
//    (std::vector<string>::iterator &args,
//     const std::vector<string>::iterator &end) {
//        // data...
//        CorkTriMesh in0;
//        CorkTriMesh in1;
//        CorkTriMesh out;
//        
//        if(args == end) { cerr << "too few args" << endl; exit(1); }
//        loadMesh(*args, &in0);
//        args++;
//        
//        if(args == end) { cerr << "too few args" << endl; exit(1); }
//        loadMesh(*args, &in1);
//        args++;
//        
//        binop(in0, in1, &out);
//        
//        if(args == end) { cerr << "too few args" << endl; exit(1); }
//        saveMesh(*args, out);
//        args++;
//        
//        freeCorkTriMesh(&out);
//        
//        delete[] in0.vertices;
//        delete[] in0.triangles;
//        delete[] in1.vertices;
//        delete[] in1.triangles;
//    };
//}


int main(int argc, char *argv[])
{
    initRand(); // that's useful
    
    if(argc < 5)
	{
        cout << "Wrong arguments: 'cork [binop] [file1] [file2] [output]" << endl;
        exit(0);
    }

	// data...
	CorkTriMesh in0;
	CorkTriMesh in1;
	CorkTriMesh out;

	loadMesh(argv[2], &in0);
	{
        bool solid = isSolid(in0);
        cout << "The mesh " << argv[2] << " is " << ((solid)? "SOLID" : "NOT SOLID") << endl;
	}

	loadMesh(argv[3], &in1);
	{
        bool solid = isSolid(in1);
        cout << "The mesh " << argv[3] << " is " << ((solid)? "SOLID" : "NOT SOLID") << endl;
	}

	const char* binop = argv[1];
	if (strcmp(binop,"union") == 0)
	{
		computeUnion(in0, in1, &out);
	}
	else if (strcmp(binop,"diff") == 0)
	{
		computeDifference(in0, in1, &out);
	}
	else if (strcmp(binop,"isct") == 0)
	{
		computeIntersection(in0, in1, &out);
	}
	else if (strcmp(binop,"xor") == 0)
	{
		computeSymmetricDifference(in0, in1, &out);
	}
	else if (strcmp(binop,"resolve") == 0)
	{
		resolveIntersections(in0, in1, &out);
	}
	else
	{
		cerr << "Unknwon binary operation!" << endl;
		return 1;
	}

	saveMesh(argv[4], out);

	freeCorkTriMesh(&out);

	delete[] in0.vertices;
	delete[] in0.triangles;
	delete[] in1.vertices;
	delete[] in1.triangles;

    return 0;
}










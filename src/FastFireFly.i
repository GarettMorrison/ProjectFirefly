%module FastFireFly // Name of module

%include "std_vector.i"


// %typemap(out) std::vector<float> {
//     npy_intp length = $1.size();
//     $result = PyArray_SimpleNew(1, &amp;length, NPY_FLOAT);
//     memcpy(PyArray_DATA((PyArrayObject*)$result),$1.data(),sizeof(float)*length);
// }

// %typemap(out) std::vector<float> {
//     npy_intp length = $1.size();
//     $result = PyArray_SimpleNew(1, &amp;length, NPY_FLOAT);
//     memcpy(PyArray_DATA((PyArrayObject*)$result),$1.data(),sizeof(float)*length);
// }


namespace std {
    %template(vectorDouble) vector<double>;
    %template(vectorDoubleDouble) vector<vector<double>>;
    // %template(transformSet) transformationSet; // Custom transformation class
};

%{
    #include "FastFireFly.h"
%}

%include "FastFireFly.h";
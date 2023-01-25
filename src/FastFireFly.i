%module FastFireFly // Name of module

%include "std_vector.i"
namespace std {
    %template(vectord) vector<double>;
}
%{
    #include "FastFireFly.h"
%}

%include "FastFireFly.h";
%module FastFireFly // Name of module

%{
    #include "FastFireFly.hh"
%}

%include "std_vector.i"

namespace std {
    %template(Vector_int) vector<unsigned int>;
    %template(Vector_double) vector<double>;
    %template(Vector_doubleDouble) vector< vector<double> >;
};


%include "FastFireFly.hh";
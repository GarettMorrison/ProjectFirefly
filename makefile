libTitle := FastFireFly

swig/_$(libTitle).so: swig/$(libTitle).o swig/$(libTitle)_wrap.o
	g++ -shared swig/$(libTitle).o swig/$(libTitle)_wrap.o -o swig/_$(libTitle).so

swig/$(libTitle).o swig/$(libTitle)_wrap.o: src/$(libTitle)_wrap.cxx swig/$(libTitle).py
	g++ -c -fpic src/$(libTitle).cpp -o swig/$(libTitle).o
	g++ -c -fpic src/$(libTitle)_wrap.cxx -I/usr/include/python3.8  -o swig/$(libTitle)_wrap.o

src/$(libTitle)_wrap.cxx swig/$(libTitle).py: src/$(libTitle).i src/$(libTitle).h src/$(libTitle).cpp
	swig -python -c++ -outdir swig src/$(libTitle).i 

clean:
	rm -f swig/$(libTitle)_wrap.o
	rm -f swig/$(libTitle)_wrap.cxx
	rm -f swig/$(libTitle).py
	rm -f swig/$(libTitle).o
	rm -f swig/_$(libTitle).so
TOPTARGETS := all clean
SUBDIRS := ./c ./pthread ./openmp ./mpi ./opencl ./openacc

$(TOPTARGETS) : $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean: $(SUBDIRS)
	rm -rf doc/html

doxygen:
	doxygen doc/Doxyfile


.PHONY: all clean $(SUBDIRS)


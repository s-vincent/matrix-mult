all:
	$(MAKE) -f Makefile.nvidia || echo "Failed to build with Nvidia offload"
	$(MAKE) -f Makefile.amd || echo "Failed to build with AMD offload"

clean:
	$(MAKE) -f Makefile.nvidia clean
	$(MAKE) -f Makefile.amd clean

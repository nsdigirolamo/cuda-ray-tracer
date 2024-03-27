SDIR := source
TDIR := test
IDIR := include
ODIR := obj

NVCC := nvcc
NVCCFLAGS := -std=c++17 -I $(IDIR)

SRCS := $(shell find $(SDIR) -name '*.cu')
OBJS := $(SRCS:$(SDIR)/%=$(ODIR)/%.o)

TSRCS := $(shell find $(TDIR) $(SDIR) -name '*.cu' -and \! -name 'main.cu' )
TOBJS := $(TSRCS:$(TDIR)/%=$(ODIR)/%.o)
TOBJS := $(TSRCS:$(SDIR)/%=$(ODIR)/%.o)

HDRS := $(shell find $(IDIR) -name '*.hpp')

.PHONY: clean

tracing: $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o tracing

testing: $(TOBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o testing

$(ODIR)/%.cpp.o: $(SDIR)/%.cpp $(HDRS) | $(ODIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(ODIR)/%.cu.o: $(SDIR)/%.cu $(HDRS) | $(ODIR)
	$(NVCC) $(NVCCFLAGS) --device-c $< -o $@

$(ODIR)/%.cpp.o: $(TDIR)/%.cpp $(HDRS) | $(ODIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(ODIR)/%.cu.o: $(TDIR)/%.cu $(HDRS) | $(ODIR)
	$(NVCC) $(NVCCFLAGS) --device-c $< -o $@

$(ODIR):
	mkdir -p $@
	mkdir -p $@/hittables
	mkdir -p $@/materials
	mkdir -p $@/utils

clean:
	rm -rf $(ODIR) a.out testing tracing *.ppm valgrind-out.* vgcore.*
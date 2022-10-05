CC := gcc
VALGRIND := valgrind

SRCDIR := src
INCLUDEDIR := include
TESTDIR := test

BUILDDIR := build
TEST_BINDIR := build/test_bin

INSTALL_PREFIX := .
BINDIR := $(INSTALL_PREFIX)/bin
LIBDIR := $(INSTALL_PREFIX)/lib

SRCEXT := c

# Reference to the find command sub-directory exclusion:
# https://stackoverflow.com/questions/4210042/how-do-i-exclude-a-directory-when-using-find
SOURCES_SHARED := $(shell find $(SRCDIR) -path $(SRCDIR)/main -prune -false -o -type f,l -name "*.$(SRCEXT)" -print )
SOURCES_MAIN := $(shell find $(SRCDIR)/main -type f,l -name *.$(SRCEXT))
TESTS := $(shell find $(TESTDIR) -type f,l -name "test_*.$(SRCEXT)")
TESTNAMES := $(patsubst $(TESTDIR)/test_%,%,$(TESTS:.$(SRCEXT)=))

TESTNAMES_CHECKPREF := $(patsubst %,check-%,$(TESTNAMES))
TESTNAMES_VALGRIND_CHECKPREF := $(patsubst %,check_valgrind-%,$(TESTNAMES))

OBJECTS_SHARED := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES_SHARED:.$(SRCEXT)=.so))
OBJECTS_MAIN := $(patsubst $(SRCDIR)/main/%,$(BUILDDIR)/%,$(SOURCES_MAIN:.$(SRCEXT)=.o))
OBJECTS_TEST := $(patsubst $(TESTDIR)/test_%,$(BUILDDIR)/test_%,$(TESTS:.$(SRCEXT)=.o))
TARGETS_MAIN := $(OBJECTS_MAIN:.o=)
TARGETS_TEST := $(patsubst $(TESTDIR)/test_%,$(TEST_BINDIR)/test_%,$(TESTS:.$(SRCEXT)=))

DEBUG_LEVEL := 1
DEBUG_OPTIMIZE_FLAGS := -g -O1 -DDEBUG_LEVEL=$(DEBUG_LEVEL)
STADNDARD_FLAGS := -std=gnu11

CFLAGS := $(STADNDARD_FLAGS) $(DEBUG_OPTIMIZE_FLAGS) -fPIC
LIB := -lm
INC := -I$(INCLUDEDIR)

.PHONY: all
all: $(TARGETS_MAIN) $(OBJECTS_SHARED)

.PHONY: install
install: $(TARGETS_MAIN) $(OBJECTS_SHARED)
	@mkdir -p $(LIBDIR)
	@mv $(OBJECTS_SHARED) $(LIBDIR)
	@mkdir -p $(BINDIR)
	@mv $(TARGETS_MAIN) $(BINDIR)

.PHONY: clean
clean:
	@echo "Cleaning ..."
	$(RM) -r $(BUILDDIR)

$(OBJECTS_SHARED): $(BUILDDIR)/%.so: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) -shared $(CFLAGS) $(INC) -o $@ $<

$(OBJECTS_MAIN): $(BUILDDIR)/%.o: $(SRCDIR)/main/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

$(OBJECTS_TEST): $(BUILDDIR)/%.o: $(TESTDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

$(TARGETS_MAIN): $(BUILDDIR)/%: $(BUILDDIR)/%.o $(OBJECTS_SHARED)
	@echo "Linking ..."
	$(CC) $^ -o $@ $(LIB)
	chmod u+x $@

$(TARGETS_TEST): $(TEST_BINDIR)/test_%: $(BUILDDIR)/test_%.o $(OBJECTS_SHARED)
	@mkdir -p $(TEST_BINDIR)
	@echo "Linking for test ..."
	$(CC) $^ $(LIB) -o $@
	chmod u+x $@

.PHONY: $(TESTNAMES_CHECKPREF)
$(TESTNAMES_CHECKPREF): check-% : $(TEST_BINDIR)/test_%
	$<

.PHONY: $(TESTNAMES_VALGRIND_CHECKPREF)
$(TESTNAMES_VALGRIND_CHECKPREF): check_valgrind-% : $(TEST_BINDIR)/test_%
	$(VALGRIND) $<

.PHONY: check-run
check-run: $(TESTNAMES_CHECKPREF)

.PHONY: check-valgrind
check-valgrind: $(TESTNAMES_VALGRIND_CHECKPREF)

.PHONY: check
check: check-run check-valgrind

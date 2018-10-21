#BENCHMARK := benchmark
#UNITTEST := unittest
#GUI := gui

#SRCDIR := src
#INC := include
#BUILDDIR := build

#BINBENCHMARK := bin/$(BENCHMARK)
#SRCBENCHMARK := $(shell find $(BENCHMARK)/$(SRCDIR) -type f -name "*.c")
#OBJBENCHMARK := $(patsubst $(LOADER)/$(SRCDIR)/%,$(BUILDDIR)/$(LOADER)/%,$(SRCLOADER:.c=.o)) 

#Definisce il nome della cartella che conterra i file in .c
SRCDIR := source

#Definisce la cartella dove ci saranno i file da includere, cioè i file .h 
INC :=-Iheader

#Definisce la cartella dove verranno messi i file .o
BUILDDIR := build

#Definisce la cartella dove verra messo l' eseguibile
TARGET := bin/CudaOfLife


#avvia il comando find da shell che cerca tutti i file .c nella cartella specificata da SRCDIR, -type f = (regular file)
SOURCES := $(shell find $(SRCDIR) -type f -name *.cpp)

#OBJECTS: trasforma il path dei file da SRCDIR e li dirotta su BUILDDIR modificando i .c in .o
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.cpp=.o))

CUSOURCES := $(shell find $(SRCDIR) -type f -name *.cu)
CUOBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(CUSOURCES:.cu=.cuo))

#Tutti i flag per le opzioni aggiuntive del compilatore (o2 è l'ottimizzatore automatico di secondo livello)   
CFLAGS=-c -Wall -pedantic -D_REENTRANT -O2
LDFLAGS=-lSDL2 -lcuda -lcudart
CC=g++ -fopenmp

CUFLAGS=-c
# -Xcompiler -fPIC
NVCC=nvcc -ccbin gcc

all: $(TARGET)

$(TARGET): $(CUOBJECTS) $(OBJECTS) 
	$(CC) $(CUOBJECTS) $(OBJECTS) $(LDFLAGS) -o $@
#$@: Produce un file di output con il nome che metto prima dei :, in questo caso prende il nome contenuto nella variabile TARGET

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) $< -o $@
#$<: include solo la prima dipendenza

$(BUILDDIR)/%.cuo: $(SRCDIR)/%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(CUFLAGS) $(INC) $< -o $@

clean:
	rm -r $(BUILDDIR) $(TARGET)


#Altri tipi di opzioni disponibili

#$?: Fornisce la lista di dipendenze che sono state modificate dall' ultima compilazione
#$^: fornisce TUTTE le dipendenze, indipendentemente dai cambiamenti recenti o meno efffettuati sulle stesse. Nomi duplicati verranno rimossi
#$: fornisce TUTTE le dipendenze, indipendentemente dai cambiamenti recenti o meno efffettuati sulle stesse.

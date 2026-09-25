#!/usr/bin/make -f
# Makefile per MultiPhiMo (DPF)

NAME = MultiPhiMo

# Versione del plugin: UNICA fonte (passata al codice con -DMPM_VERSION_*, usata dal nome
# dello zip di release). Per una nuova release basta cambiare questi tre numeri.
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 0
VERSION = $(VERSION_MAJOR).$(VERSION_MINOR).$(VERSION_PATCH)

FILES_DSP = \
	PluginMultiPhiMo.cpp \
	../../oscpack/osc/OscTypes.cpp \
	../../oscpack/osc/OscReceivedElements.cpp \
	../../oscpack/ip/IpEndpointName.cpp \
	../../oscpack/ip/posix/NetworkingUtils.cpp \
	../../oscpack/ip/posix/UdpSocket.cpp

# oscpack e' vendorizzato in native/oscpack (round 3, solo lato ricezione UDP: i file
# posix/ bastano per macOS/Linux, la variante ip/win32/*.cpp andra' aggiunta quando si
# buildera' su Windows -- vedi claude/vst3_native_stato.md).

# GUI (punto 8A, round 1) -- DGL/NanoVG, nessun toolkit esterno/asset bitmap (deciso con
# l'utente 2026-09-23). DISTRHO_PLUGIN_HAS_UI=1 in DistrhoPluginInfo.h attiva la build.
# Ripristinata 2026-09-23 dopo un tentativo di GUI generica host (troppi parametri
# visibili, vedi stato doc) -- questi file non erano stati toccati nel frattempo.
FILES_UI = \
	UIMultiPhiMo.cpp

CXXFLAGS += -I../../oscpack
CXXFLAGS += -DMPM_VERSION_MAJOR=$(VERSION_MAJOR) -DMPM_VERSION_MINOR=$(VERSION_MINOR) -DMPM_VERSION_PATCH=$(VERSION_PATCH)
# Il motore nativo (Engine.hpp/VoiceEngine.hpp/...) usa variabili inline constexpr e altre
# feature C++17 -- Makefile.base.mk fissa -std=gnu++11 prima di $(CXXFLAGS), l'override qui
# (ultimo -std sulla riga di comando vince) lo alza a C++17 (2026-09-22, punto 7d).
CXXFLAGS += -std=c++17
# DPF (Makefile.base.mk) forza -ffast-math in BASE_OPTS per tutti i plugin -- questo
# rompe la semantica IEEE di isnan()/isinf() (assunta 'mai NaN/Inf' dall'ottimizzatore,
# che puo' quindi eliminare i controlli come se fossero sempre falsi). Il motore usa
# isnan come sentinella (drain OSC in PluginMultiPhiMo.cpp, bestDist in KnnCorpus.hpp)
# -- causa diagnosticata del bug 'nessun cambiamento in DAW' 2026-09-22: il drain OSC
# sovrascriveva ogni parametro con un valore NaN ad ogni blocco audio perche' il check
# 'if not isnan' risultava sempre vero. L'override (ultimo -f vince) ripristina la
# semantica IEEE corretta.
CXXFLAGS += -fno-fast-math

include ../../dpf/Makefile.plugins.mk

# Dati del motore DENTRO il bundle (2026-09-25, punto 8D): pesi MDN, corpus KNN e corpus
# del selettore in <bundle>/Contents/Resources/data, letti dal plugin con getResourcePath
# di DPF. Ricopiati a ogni `make vst3` (cartella ricreata: nessun file stantio).
# Esclusi scale-library/ e i CSV (non usati dal codice).
DATA_SRC = ../../data
DATA_DST = $(TARGET_DIR)/$(NAME).vst3/Contents/Resources/data

vst3: vst3-data

vst3-data: $(vst3)
	rm -rf $(DATA_DST)
	mkdir -p $(DATA_DST)/weights
	cp $(DATA_SRC)/knn_corpus.bin $(DATA_SRC)/selector_corpus.bin $(DATA_DST)/
	cp $(DATA_SRC)/weights/*.bin $(DATA_DST)/weights/

# Release macOS (Apple Silicon, macOS 11+), 2026-09-25: build da zero, firma ad hoc
# gratuita, zip con plugin + README + licenze. Gli oggetti di oscpack vanno cancellati a
# mano perche' DPF li scrive accanto ai sorgenti, non in native/build.
RELEASE_DIR = ../../bin
RELEASE_ZIP = $(NAME)-$(VERSION)-macOS-arm64.zip

release:
	find ../../oscpack \( -name "*.cpp.o" -o -name "*.cpp.d" \) -delete
	rm -rf ../../build/$(NAME) ../../dpf/build $(RELEASE_DIR)/$(NAME).vst3 $(RELEASE_DIR)/dist $(RELEASE_DIR)/$(RELEASE_ZIP)
	MACOSX_DEPLOYMENT_TARGET=11.0 $(MAKE) vst3
	codesign --force --deep -s - $(RELEASE_DIR)/$(NAME).vst3
	codesign --verify --deep --strict $(RELEASE_DIR)/$(NAME).vst3
	mkdir -p $(RELEASE_DIR)/dist/$(NAME)
	cp -R $(RELEASE_DIR)/$(NAME).vst3 ../../README.md ../../LICENSE ../../THIRD_PARTY_LICENSES.md $(RELEASE_DIR)/dist/$(NAME)/
	cd $(RELEASE_DIR)/dist && ditto -c -k --norsrc --noextattr --keepParent $(NAME) ../$(RELEASE_ZIP)
	rm -rf $(RELEASE_DIR)/dist
	@echo "Release pronta: native/bin/$(RELEASE_ZIP)"

.PHONY: vst3-data release

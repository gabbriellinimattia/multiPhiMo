#!/usr/bin/make -f
# Makefile per MultiPhiMo (DPF)

NAME = MultiPhiMo

# Versione del plugin: UNICA fonte (passata al codice con -DMPM_VERSION_*, usata dal nome
# dello zip di release). Per una nuova release basta cambiare questi tre numeri.
VERSION_MAJOR = 0
VERSION_MINOR = 2
VERSION_PATCH = 0
VERSION = $(VERSION_MAJOR).$(VERSION_MINOR).$(VERSION_PATCH)

FILES_DSP = \
	PluginMultiPhiMo.cpp \
	../../oscpack/osc/OscTypes.cpp \
	../../oscpack/osc/OscReceivedElements.cpp \
	../../oscpack/ip/IpEndpointName.cpp \
	../../oscpack/ip/posix/NetworkingUtils.cpp \
	../../oscpack/ip/posix/UdpSocket.cpp

FILES_UI = \
	UIMultiPhiMo.cpp

CXXFLAGS += -I../../oscpack
CXXFLAGS += -DMPM_VERSION_MAJOR=$(VERSION_MAJOR) -DMPM_VERSION_MINOR=$(VERSION_MINOR) -DMPM_VERSION_PATCH=$(VERSION_PATCH)
# C++17 (Makefile.base.mk fissa -std=gnu++11 prima di $(CXXFLAGS): l'ultimo -std vince)
CXXFLAGS += -std=c++17
# DPF forza -ffast-math: ripristina la semantica IEEE (vedi claude/vst3_native_stato.md, bug B punto 7d)
CXXFLAGS += -fno-fast-math

include ../../dpf/Makefile.plugins.mk

# Dati del motore DENTRO il bundle (<bundle>/Contents/Resources/data), letti con getResourcePath.
# 2026-09-28 (0.2.0): surrogati v5 (native/data/surrogate, export_surrogate.py) + corpus del selettore;
# pesi MDN e corpus KNN non piu' usati.
DATA_SRC = ../../data
DATA_DST = $(TARGET_DIR)/$(NAME).vst3/Contents/Resources/data

vst3: vst3-data

vst3-data: $(vst3)
	rm -rf $(DATA_DST)
	mkdir -p $(DATA_DST)/surrogate
	cp $(DATA_SRC)/selector_corpus.bin $(DATA_DST)/
	cp $(DATA_SRC)/surrogate/*.bin $(DATA_DST)/surrogate/

# Release macOS (Apple Silicon, macOS 11+): build da zero, firma ad hoc, zip con plugin + README + licenze.
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

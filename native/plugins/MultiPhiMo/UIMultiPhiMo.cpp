/*
 * MultiPhiMo -- GUI (punto 8A).
 * Round 1 (2026-09-23): Mode toggle (Agent/Manual), selettore Exciter, selettore
 * Resonator (nomi reali, dropdown al click), 15 slider descrittori.
 * Round 2 (2026-09-23): slot manuali con nomi/range REALI per l'eccitatore/risonatore
 * attivo (phimo::agentSpecs(), ParamRanges.hpp) -- vista esclusiva col toggle Mode:
 * Agent mostra i 15 descrittori, Manual mostra i controlli diretti del synth (fino a
 * 9 slot eccitatore + 10 slot risonatore, nascosti oltre il conteggio reale
 * dell'agente attivo). Il parametro host per ogni slot resta normalizzato 0-1 fisso
 * (design deliberato, vedi claude/vst3_native_stato.md -- l'automazione host non deve
 * dipendere da quale eccitatore e' selezionato): la conversione fisico<->normalizzato
 * (lineare o log10, stessa formula di Slider::valueToFraction/fractionToValue in
 * UiWidgets.hpp) avviene qui in GUI in base alla ParamSpec dell'agente corrente.
 * Round 3.1 (2026-09-23): pulsante "Play" (UI::sendNote, hold-to-sustain) -- nessuna
 * modifica lato DSP: run() maschera lo status MIDI con 0xF0 (canale ignorato), quindi un
 * note-on/off inviato da sendNote() attraversa esattamente lo stesso percorso gia'
 * validato dal vivo per una tastiera MIDI reale.
 * Round 3.2 (2026-09-23): indicatore "OSC in ascolto" via state plugin (stateChanged()),
 * NON un parametro host. Il valore arriva alla UI quando si (ri)connette all'editor
 * (DISTRHO_PLUGIN_WANT_FULL_STATE fa rileggere lo stato vero dal DSP a quel momento,
 * vedi DistrhoPluginInfo.h/PluginMultiPhiMo.cpp) -- sufficiente perche' il bind OSC e'
 * fisso dopo il costruttore del plugin, non cambia mai mentre il plugin gira.
 * Round 3.3a (2026-09-23): Scale/tuning built-in (edo12/24/31, perfect, harmonic) -- 3
 * nuovi state (scale_active/scale_name/a4, ParamLayout.hpp), SCRIVIBILI dalla UI stavolta
 * (non solo letti come osc_active). Il canale MIDI del pulsante Play e' stato spostato a
 * kPlayMidiChannel (riservato, vedi ParamLayout.hpp): su quel canale la nota non
 * sovrascrive mai pitch/freq, su qualunque altro canale (MIDI reale) lo fa se Scale e'
 * attivo -- decisione esplicita dell'utente, quantizzazione vera nel worker thread
 * (VoiceEngine.hpp/TuningQuantizer.hpp), mai qui in UI ne' nel thread audio.
 * NON ancora in questa GUI (rimandato al round 3.3b, concordato con l'utente): import
 * `.scl` custom.
 * Vedi claude/vst3_native_stato.md.
 */

#include "DistrhoUI.hpp"
#include "UiWidgets.hpp"
#include "ParamRanges.hpp"
#include "TuningQuantizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

START_NAMESPACE_DISTRHO

#include "ParamLayout.hpp"

using DGL_NAMESPACE::Slider;
using DGL_NAMESPACE::Toggle;
using DGL_NAMESPACE::CycleChoice;
using DGL_NAMESPACE::DropdownPopup;
using DGL_NAMESPACE::PlayButton;
using DGL_NAMESPACE::Color;

// ---------------------------------------------------------------------------------

static constexpr uint kUiWidth  = 760;
static constexpr uint kUiHeight = 500;  // 2026-09-25: +40 per la riga Morph (y=468)

// Pulsante Play (round 3.1): note/canale arbitrari, il pitch resta governato SOLO dal
// descrittore target (Agent) o dallo slot "freq" (Manual), mai dal numero di nota --
// stesso principio gia' documentato per il MIDI reale (punto 7d). Velocity 100 ->
// gain=100/116=0.862 (formula decisa al punto 7d), entro il margine gia' validato dal
// vivo per note reali (picco massimo osservato 0.857, limiter_ceiling=0.85).
static constexpr uint8_t kPlayButtonNote     = 60;
static constexpr uint8_t kPlayButtonVelocity = 100;

// Conversione fisico<->normalizzato per gli slot manuali -- il parametro host resta
// normalizzato 0-1 a prescindere dall'agente attivo (vedi commento in testa), la Slider
// invece lavora sempre in valore fisico (lo..hi) come gia' fanno i 15 descrittori.
// phimo::paramSpecDenormalize/paramSpecNormalize (ParamRanges.hpp, punto 8A round 2,
// 2026-09-23) -- STESSA funzione usata dal DSP (VoiceEngine.hpp, Mode=Manual) per non
// avere due formule che possono divergere tra GUI e motore audio.
using phimo::paramSpecDenormalize;
using phimo::paramSpecNormalize;

class MultiPhiMoUI : public UI
{
public:
    MultiPhiMoUI()
        : UI(kUiWidth, kUiHeight),
          fCurrentExciterSpec(nullptr), fCurrentResonatorSpec(nullptr),
          fOscActive(false)
    {
        // etichette exciter/resonator prese da kExciterEnum/kResonatorEnum (ParamLayout.hpp,
        // condivise col DSP) -- copiate in array membro perche' CycleChoice memorizza solo
        // il puntatore, deve restare valido per tutta la vita del widget.
        for (uint32_t i = 0; i < 10; ++i)
            fExciterLabels[i] = kExciterEnum[i].label;
        for (uint32_t i = 0; i < 7; ++i)
            fResonatorLabels[i] = kResonatorEnum[i].label;

        for (uint32_t s = 0; s < 9; ++s)
            fExciterParamRaw[s] = 0.5f;
        for (uint32_t s = 0; s < 10; ++s)
            fResonatorParamRaw[s] = 0.5f;

        // -------- riga superiore: Mode / Exciter / Resonator --------
        fModeToggle = new Toggle(this);
        fModeToggle->setAbsolutePos(8, 8);
        fModeToggle->setSize(120, 32);
        fModeToggle->setLabel("Mode: Agent");
        fModeToggle->onToggled = [this](bool checked)
        {
            fModeToggle->setLabel(checked ? "Mode: Manual" : "Mode: Agent");
            updateViewForMode(checked);
            const float v = checked ? 1.0f : 0.0f;
            editParameter(kParameterMode, true);
            setParameterValue(kParameterMode, v);
            editParameter(kParameterMode, false);
        };

        fExciterChoice = new CycleChoice(this);
        fExciterChoice->setAbsolutePos(136, 8);
        fExciterChoice->setSize(160, 32);  // 200->160 (2026-09-25) per far posto a Gain
        fExciterChoice->setTitle("Exciter");
        fExciterChoice->setOptions(fExciterLabels, 10);
        fExciterChoice->onChanged = [this](uint32_t idx)
        {
            updateManualExciterSpec();
            editParameter(kParameterExciterSelect, true);
            setParameterValue(kParameterExciterSelect, static_cast<float>(idx));
            editParameter(kParameterExciterSelect, false);
        };

        fResonatorChoice = new CycleChoice(this);
        fResonatorChoice->setAbsolutePos(304, 8);
        fResonatorChoice->setSize(160, 32);
        fResonatorChoice->setTitle("Resonator");
        fResonatorChoice->setOptions(fResonatorLabels, 7);
        fResonatorChoice->onChanged = [this](uint32_t idx)
        {
            updateManualResonatorSpec();
            editParameter(kParameterResonatorSelect, true);
            setParameterValue(kParameterResonatorSelect, static_cast<float>(idx));
            editParameter(kParameterResonatorSelect, false);
        };

        // Play (round 3.1): sempre visibile in entrambe le viste (Agent/Manual), stesso
        // trattamento di Mode/Exciter/Resonator sopra -- non fa parte dello swap esclusivo
        // di updateViewForMode. A destra del selettore Resonator (che termina a x=544).
        fPlayButton = new PlayButton(this);
        fPlayButton->setAbsolutePos(472, 8);
        fPlayButton->setSize(80, 32);
        fPlayButton->setLabel("Play");
        // Round 3.3a: canale riservato (kPlayMidiChannel, ParamLayout.hpp) -- run() lo usa
        // per NON sovrascrivere mai pitch/freq da questo pulsante anche se Scale e' attivo
        // (deciso con l'utente 2026-09-23), a differenza di una tastiera MIDI reale.
        fPlayButton->onPress   = [this]() { sendNote(kPlayMidiChannel, kPlayButtonNote, kPlayButtonVelocity); };
        fPlayButton->onRelease = [this]() { sendNote(kPlayMidiChannel, kPlayButtonNote, 0); };

        // Gain master (2026-09-25, porting di gui.py slider "Gain" 0-1.5): parametro host
        // kParameterGain, sempre visibile come Play. x=560..700, OSC spostato a x>=712.
        fGainSlider = new Slider(this);
        fGainSlider->setAbsolutePos(560, 8);
        fGainSlider->setSize(140, 32);
        fGainSlider->setName("Gain");
        fGainSlider->setRange(0.0f, 1.5f, false);
        fGainSlider->setValueQuiet(0.4f);
        fGainSlider->onChanged = [this](float v)
        {
            editParameter(kParameterGain, true);
            setParameterValue(kParameterGain, v);
            editParameter(kParameterGain, false);
        };

        // -------- 15 slider descrittori, 2 colonne (8 + 7) -- vista "Agent" --------
        static constexpr uint kSliderW = 364;
        static constexpr uint kSliderH = 34;
        static constexpr uint kRowGap  = 4;
        static constexpr uint kTopY    = 52;
        static constexpr uint kColX[2] = {8, 384};

        for (uint32_t d = 0; d < 15; ++d)
        {
            const uint32_t col = d / 8;
            const uint32_t row = d % 8;

            Slider* const s = new Slider(this);
            s->setAbsolutePos(kColX[col], kTopY + row * (kSliderH + kRowGap));
            s->setSize(kSliderW, kSliderH);
            s->setName(kDescriptorNames[d]);
            s->setRange(kDescriptorRanges[d].lo, kDescriptorRanges[d].hi, kDescriptorRanges[d].logScale);
            s->setValueQuiet(kDescriptorRanges[d].def);
            fDescriptorValue[d] = kDescriptorRanges[d].def;

            const uint32_t paramIndex = kParameterDescriptorFirst + d;
            s->onDragStart = [this, paramIndex]() { editParameter(paramIndex, true); };
            s->onDragEnd   = [this, paramIndex]() { editParameter(paramIndex, false); };
            s->onChanged   = [this, paramIndex, d](float v) { fDescriptorValue[d] = v; setParameterValue(paramIndex, v); };

            fSliders[d] = s;
        }

        // Morph spettrale (2026-09-25): riga in fondo (y=468), visibile in entrambe le viste.
        fMorphToggle = new Toggle(this);
        fMorphToggle->setAbsolutePos(8, 468);
        fMorphToggle->setSize(110, 26);
        fMorphToggle->setLabel("Morph: Off");
        fMorphToggle->onToggled = [this](bool checked)
        {
            fMorphToggle->setLabel(checked ? "Morph: On" : "Morph: Off");
            editParameter(kParameterMorph, true);
            setParameterValue(kParameterMorph, checked ? 1.0f : 0.0f);
            editParameter(kParameterMorph, false);
        };

        // Auto pair (2026-09-25): stessa riga del Morph, solo vista Agent.
        fAutoPairToggle = new Toggle(this);
        fAutoPairToggle->setAbsolutePos(126, 468);
        fAutoPairToggle->setSize(130, 26);
        fAutoPairToggle->setLabel("Auto pair: Off");
        fAutoPairToggle->onToggled = [this](bool checked)
        {
            fAutoPairToggle->setLabel(checked ? "Auto pair: On" : "Auto pair: Off");
            editParameter(kParameterAutoPair, true);
            setParameterValue(kParameterAutoPair, checked ? 1.0f : 0.0f);
            editParameter(kParameterAutoPair, false);
        };

        // Audio-in (2026-09-25, porting di audio_input.py/gui.py): toggle + Smoothing nella
        // cella libera della vista Agent (colonna 2, riga 8: i descrittori sono 8+7).
        // Nascosti in Manual (stessa area degli slot risonatore).
        {
            const uint y = kTopY + 7 * (kSliderH + kRowGap);
            fAudioInToggle = new Toggle(this);
            fAudioInToggle->setAbsolutePos(kColX[1], y);
            fAudioInToggle->setSize(118, 32);
            fAudioInToggle->setLabel("Audio In: Off");
            fAudioInToggle->onToggled = [this](bool checked)
            {
                fAudioInToggle->setLabel(checked ? "Audio In: On" : "Audio In: Off");
                editParameter(kParameterAudioIn, true);
                setParameterValue(kParameterAudioIn, checked ? 1.0f : 0.0f);
                editParameter(kParameterAudioIn, false);
            };

            fSmoothingSlider = new Slider(this);
            fSmoothingSlider->setAbsolutePos(kColX[1] + 126, y);
            fSmoothingSlider->setSize(kSliderW - 126, kSliderH);
            fSmoothingSlider->setName("Smoothing (ms)");
            fSmoothingSlider->setRange(20.0f, 1000.0f, false);
            fSmoothingSlider->setValueQuiet(200.0f);
            fSmoothingSlider->onDragStart = [this]() { editParameter(kParameterSmoothing, true); };
            fSmoothingSlider->onDragEnd   = [this]() { editParameter(kParameterSmoothing, false); };
            fSmoothingSlider->onChanged   = [this](float v) { setParameterValue(kParameterSmoothing, v); };
        }

        // -------- slot manuali eccitatore (fino a 9) / risonatore (fino a 10) --------
        // Stessa griglia a 2 colonne dei descrittori, colonna 0 = eccitatore, colonna 1 =
        // risonatore. Nome/range REALI assegnati da updateManualExciterSpec/
        // updateManualResonatorSpec (chiamate sotto, dopo la creazione), non qui: qui solo
        // i widget vuoti, altrimenti sarebbero configurati due volte.
        for (uint32_t s = 0; s < 9; ++s)
        {
            Slider* const sl = new Slider(this);
            sl->setAbsolutePos(kColX[0], kTopY + s * (kSliderH + kRowGap));
            sl->setSize(kSliderW, kSliderH);
            sl->onDragStart = [this, s]() { editParameter(kParameterExciterParamFirst + s, true); };
            sl->onDragEnd   = [this, s]() { editParameter(kParameterExciterParamFirst + s, false); };
            sl->onChanged   = [this, s](float physicalValue)
            {
                if (fCurrentExciterSpec == nullptr || s >= fCurrentExciterSpec->size())
                    return;
                const float norm = paramSpecNormalize(physicalValue, (*fCurrentExciterSpec)[s]);
                fExciterParamRaw[s] = norm;
                setParameterValue(kParameterExciterParamFirst + s, norm);
            };
            fExciterParamSliders[s] = sl;
        }

        for (uint32_t s = 0; s < 10; ++s)
        {
            Slider* const sl = new Slider(this);
            sl->setAbsolutePos(kColX[1], kTopY + s * (kSliderH + kRowGap));
            sl->setSize(kSliderW, kSliderH);
            sl->onDragStart = [this, s]() { editParameter(kParameterResonatorParamFirst + s, true); };
            sl->onDragEnd   = [this, s]() { editParameter(kParameterResonatorParamFirst + s, false); };
            sl->onChanged   = [this, s](float physicalValue)
            {
                if (fCurrentResonatorSpec == nullptr || s >= fCurrentResonatorSpec->size())
                    return;
                const float norm = paramSpecNormalize(physicalValue, (*fCurrentResonatorSpec)[s]);
                fResonatorParamRaw[s] = norm;
                setParameterValue(kParameterResonatorParamFirst + s, norm);
            };
            fResonatorParamSliders[s] = sl;
        }

        // -------- riga Scale/tuning (round 3.3a) -- sempre visibile, entrambe le viste --------
        // y=428, h=32: spazio libero sotto l'ultima riga degli slot manuali risonatore
        // (10 slot: 52 + 9*38 = 394, +34 = 428 -- esattamente il margine lasciato dai 460px
        // di altezza finestra fissati al round 2 per il risonatore piu' esteso, chaotic).
        static constexpr uint kScaleRowY = 428;

        fScaleToggle = new Toggle(this);
        fScaleToggle->setAbsolutePos(8, kScaleRowY);
        fScaleToggle->setSize(110, 32);
        fScaleToggle->setLabel("Scale: Off");
        fScaleToggle->onToggled = [this](bool checked)
        {
            fScaleToggle->setLabel(checked ? "Scale: On" : "Scale: Off");
            setState(kStateKeyScaleActive, checked ? "1" : "0");
        };

        for (uint32_t i = 0; i < phimo::kScaleCount; ++i)
            fScaleLabels[i] = phimo::scaleIndexToName(static_cast<int>(i));

        fScaleChoice = new CycleChoice(this);
        fScaleChoice->setAbsolutePos(126, kScaleRowY);
        fScaleChoice->setSize(130, 32);
        fScaleChoice->setTitle("Scala");
        fScaleChoice->setOptions(fScaleLabels, phimo::kScaleCount);
        // Bug osservato dal vivo (2026-09-23): a y=428 su una finestra di 460px il popup
        // (di default sotto il widget) si apriva fuori dall'area visibile -- vedi
        // CycleChoice::setOpenUpward in UiWidgets.hpp. Exciter/Resonator restano invariati
        // (sono in cima alla finestra, il popup verso il basso ci sta).
        fScaleChoice->setOpenUpward(true);
        fScaleChoice->onChanged = [this](uint32_t idx)
        {
            setState(kStateKeyScaleName, phimo::scaleIndexToName(static_cast<int>(idx)));
        };

        fA4Slider = new Slider(this);
        fA4Slider->setAbsolutePos(264, kScaleRowY);
        fA4Slider->setSize(200, 32);
        fA4Slider->setName("A4");
        fA4Slider->setRange(400.0f, 480.0f, false);
        fA4Slider->setValueQuiet(440.0f);
        fA4Slider->onChanged = [this](float v)
        {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.2f", static_cast<double>(v));
            setState(kStateKeyA4, buf);
        };

        // Round 3.3b: bottone "Load .scl..." -- UI::requestStateFile(key). VERIFICATO nel
        // sorgente DPF: il backend VST3 passa nullptr come fileRequestCallback ("TODO file
        // request", DistrhoUIVST3.cpp), ma UI::PrivateData::fileRequestCallback ricade sul
        // browser file di DGL (DISTRHO_UI_FILE_BROWSER, default 1) e, a scelta fatta,
        // PluginWindow::onFileSelected chiama setState(key,path) (DSP: ignorato) e
        // stateChanged(key,path) -- gestito sotto in stateChanged().
        fSclButton = new PlayButton(this);
        fSclButton->setAbsolutePos(472, kScaleRowY);
        fSclButton->setSize(118, 32);
        fSclButton->setLabel("Load .scl...");
        fSclButton->onPress = [this]() { requestStateFile(kStateKeySclImport); };

        // Popup dropdown condiviso -- va creato PER ULTIMO: WidgetPrivateData itera i
        // subwidget in ordine inverso di creazione per gli eventi mouse (vedi nota in
        // UiWidgets.hpp/DropdownPopup), quindi solo cosi' resta sopra a tutto il resto
        // (sliders compresi) mentre e' aperto.
        fPopup = new DropdownPopup(this);
        fExciterChoice->setPopup(fPopup);
        fResonatorChoice->setPopup(fPopup);
        fScaleChoice->setPopup(fPopup);

        // Configura gli slot manuali sull'agente di default (indice 0 = "bow"/"bar", stesso
        // default di fExciterChoice/fResonatorChoice) e imposta la vista iniziale su Agent
        // (Mode di default = 0, vedi kModeEnum) -- prima di questo i nuovi slider sarebbero
        // visibili di default (Widget parte visible=true) sovrapposti ai descrittori.
        updateManualExciterSpec();
        updateManualResonatorSpec();
        updateViewForMode(false);

        // Ridimensionabile (punto 8A round 1, 2026-09-23): setGeometryConstraints va
        // chiamata per ULTIMA, a tutti i widget gia' creati -- un tentativo precedente con
        // questa chiamata come PRIMA istruzione del costruttore causava un crash in Reaper
        // (vedi claude/vst3_native_stato.md). true,true = mantieni proporzioni + scala il
        // contenuto (richiede NanoTopLevelWidget* come tipo dei parent dei widget figli,
        // vedi UiWidgets.hpp).
        setGeometryConstraints(kUiWidth, kUiHeight, true, true);
    }

protected:
    // ---------------------------------------------------------------
    // onNanoDisplay e' puro virtuale in NanoTopLevelWidget (NanoVG.hpp) -- qui solo
    // lo sfondo, i controlli disegnano se stessi nei propri onNanoDisplay (UiWidgets.hpp).
    void onNanoDisplay() override
    {
        beginPath();
        fillColor(Color(28, 28, 32));
        rect(0, 0, getWidth(), getHeight());
        fill();
        closePath();

        // Indicatore "OSC in ascolto" (round 3.2) -- disegnato direttamente qui (nessun
        // widget interattivo, solo lettura) a destra del pulsante Play (che finisce a
        // x=552+96=648). Pallino verde=bind riuscito, grigio=non attivo (porta occupata o
        // altro errore, vedi PluginMultiPhiMo.cpp::startOsc).
        beginPath();
        fillColor(fOscActive ? Color(90, 200, 110) : Color(90, 90, 96));
        circle(716.0f, 24.0f, 6.0f);  // spostato per Gain (2026-09-25)
        fill();
        closePath();

        fontSize(12.0f);
        textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
        fillColor(Color(200, 200, 200));
        text(726.0f, 24.0f, "OSC", nullptr);

        // Round 3.3b: nome del .scl caricato (o errore), a destra del bottone Load.
        if (!fSclLabel.empty())
            text(598.0f, 444.0f, fSclLabel.c_str(), nullptr);
    }

    // ---------------------------------------------------------------
    // DSP -> UI: riflette lo stato host sui widget SENZA richiamare i loro
    // callback (altrimenti si rimanderebbe subito il valore all'host in loop).

    void parameterChanged(uint32_t index, float value) override
    {
        if (index == kParameterMode)
        {
            const bool checked = value >= 0.5f;
            fModeToggle->setCheckedQuiet(checked);
            fModeToggle->setLabel(checked ? "Mode: Manual" : "Mode: Agent");
            updateViewForMode(checked);
            return;
        }

        if (index == kParameterExciterSelect)
        {
            fExciterChoice->setIndexQuiet(static_cast<uint32_t>(value + 0.5f));
            updateManualExciterSpec();
            return;
        }

        if (index == kParameterGain)
        {
            fGainSlider->setValueQuiet(value);
            return;
        }

        if (index == kParameterAudioIn)
        {
            const bool checked = value >= 0.5f;
            fAudioInToggle->setCheckedQuiet(checked);
            fAudioInToggle->setLabel(checked ? "Audio In: On" : "Audio In: Off");
            repaint();
            return;
        }

        if (index == kParameterAutoPair)
        {
            const bool checked = value >= 0.5f;
            fAutoPairToggle->setCheckedQuiet(checked);
            fAutoPairToggle->setLabel(checked ? "Auto pair: On" : "Auto pair: Off");
            repaint();
            return;
        }

        if (index == kParameterMorph)
        {
            const bool checked = value >= 0.5f;
            fMorphToggle->setCheckedQuiet(checked);
            fMorphToggle->setLabel(checked ? "Morph: On" : "Morph: Off");
            repaint();
            return;
        }

        if (index == kParameterSmoothing)
        {
            fSmoothingSlider->setValueQuiet(value);
            return;
        }

        if (index == kParameterResonatorSelect)
        {
            fResonatorChoice->setIndexQuiet(static_cast<uint32_t>(value + 0.5f));
            updateManualResonatorSpec();
            return;
        }

        if (index >= kParameterDescriptorFirst && index <= kParameterDescriptorLast)
        {
            fDescriptorValue[index - kParameterDescriptorFirst] = value;
            fSliders[index - kParameterDescriptorFirst]->setValueQuiet(value);
            return;
        }

        if (index >= kParameterExciterParamFirst && index <= kParameterExciterParamLast)
        {
            const uint32_t s = index - kParameterExciterParamFirst;
            fExciterParamRaw[s] = value;
            if (fCurrentExciterSpec != nullptr && s < fCurrentExciterSpec->size())
                fExciterParamSliders[s]->setValueQuiet(paramSpecDenormalize(value, (*fCurrentExciterSpec)[s]));
            return;
        }

        if (index >= kParameterResonatorParamFirst && index <= kParameterResonatorParamLast)
        {
            const uint32_t s = index - kParameterResonatorParamFirst;
            fResonatorParamRaw[s] = value;
            if (fCurrentResonatorSpec != nullptr && s < fCurrentResonatorSpec->size())
                fResonatorParamSliders[s]->setValueQuiet(paramSpecDenormalize(value, (*fCurrentResonatorSpec)[s]));
            return;
        }
    }

    // Round 3.2/3.3a: spinte dal DSP quando la UI si connette (mai "live" mentre resta
    // aperta per osc_active -- non serve, il bind OSC e' fisso dopo il costruttore, vedi
    // nota in testa al file; per le 3 chiavi Scale invece SI aggiorna anche mentre la UI e'
    // gia' aperta, se un'altra istanza UI o il ripristino di un progetto le cambia altrove
    // -- setCheckedQuiet/setIndexQuiet/setValueQuiet evitano di rimandare il valore
    // all'host in loop, stesso principio di parameterChanged() sopra).
    void stateChanged(const char* key, const char* value) override
    {
        if (std::strcmp(key, kStateKeyOscActive) == 0)
        {
            fOscActive = (value != nullptr && value[0] == '1');
            repaint();
        }
        else if (std::strcmp(key, kStateKeyScaleActive) == 0)
        {
            const bool checked = (value != nullptr && value[0] == '1');
            fScaleToggle->setCheckedQuiet(checked);
            fScaleToggle->setLabel(checked ? "Scale: On" : "Scale: Off");
        }
        else if (std::strcmp(key, kStateKeyScaleName) == 0)
        {
            fScaleChoice->setIndexQuiet(static_cast<uint32_t>(phimo::scaleNameToIndex(value)));
        }
        else if (std::strcmp(key, kStateKeyA4) == 0)
        {
            fA4Slider->setValueQuiet(static_cast<float>(std::atof(value)));
        }
        else if (std::strcmp(key, kStateKeySclImport) == 0)
        {
            // Risposta di requestStateFile: path scelto. Valore vuoto = push iniziale di
            // FULL_STATE (default), da ignorare. Lettura+parsing qui in UI (thread UI, mai
            // audio); il DSP riceve solo la forma canonica gia' validata (scl_data).
            if (value == nullptr || value[0] == '\0')
                return;
            std::string text;
            {
                std::ifstream f(value, std::ios::binary);
                if (f)
                {
                    std::ostringstream ss;
                    ss << f.rdbuf();
                    text = ss.str();
                }
            }
            phimo::CustomScale cs;
            if (!text.empty() && text.size() <= 65536 && phimo::parseScl(text, cs))
            {
                const std::string canon = phimo::serializeCustomScale(cs);
                setState(kStateKeySclFile, value);
                setState(kStateKeySclData, canon.c_str());
                setState(kStateKeyScaleName, phimo::scaleIndexToName(phimo::kScaleCustom));
                fScaleChoice->setIndexQuiet(static_cast<uint32_t>(phimo::kScaleCustom));
                setSclLabel(value);
            }
            else
            {
                fSclLabel = "file .scl non valido";
                repaint();
            }
        }
        else if (std::strcmp(key, kStateKeySclFile) == 0)
        {
            // Ripristino progetto / riconnessione UI: solo etichetta, nessuna rilettura del file.
            setSclLabel(value);
        }
        else if (std::strcmp(key, kStateKeySclData) == 0)
        {
            if (value == nullptr || value[0] == '\0')
            {
                fSclLabel.clear();
                repaint();
            }
        }
    }

    void setSclLabel(const char* path)
    {
        std::string s = (path != nullptr) ? path : "";
        const size_t sep = s.find_last_of("/\\");
        if (sep != std::string::npos) s = s.substr(sep + 1);
        if (s.size() > 22) s = s.substr(0, 19) + "...";
        fSclLabel = s;
        repaint();
    }

private:
    Toggle* fModeToggle;
    CycleChoice* fExciterChoice;
    CycleChoice* fResonatorChoice;
    PlayButton* fPlayButton;
    Slider* fGainSlider;
    Toggle* fAudioInToggle;    // 2026-09-25 audio-in
    Toggle* fMorphToggle;      // 2026-09-25 morph spettrale
    Toggle* fAutoPairToggle;   // 2026-09-25 auto pair (solo Agent)
    Slider* fSmoothingSlider;
    Slider* fSliders[15];
    float fDescriptorValue[15];  // valore host corrente (non clampato al range GUI)
    Slider* fExciterParamSliders[9];
    Slider* fResonatorParamSliders[10];
    DropdownPopup* fPopup;

    // Round 3.3a
    Toggle* fScaleToggle;
    CycleChoice* fScaleChoice;
    Slider* fA4Slider;
    PlayButton* fSclButton;   // round 3.3b: "Load .scl..."
    std::string fSclLabel;
    const char* fScaleLabels[phimo::kScaleCount];

    const char* fExciterLabels[10];
    const char* fResonatorLabels[7];

    float fExciterParamRaw[9];
    float fResonatorParamRaw[10];
    const std::vector<phimo::ParamSpec>* fCurrentExciterSpec;
    const std::vector<phimo::ParamSpec>* fCurrentResonatorSpec;
    bool fOscActive;

    // Mostra/nasconde i 15 descrittori (Agent) contro gli slot manuali (Manual) -- vista
    // esclusiva, mai entrambe visibili insieme (richiesto dall'utente 2026-09-23: "in mode
    // manual dovrei vedere i controlli diretti dei synth", non i descrittori assieme).
    void updateViewForMode(bool manual)
    {
        for (uint32_t d = 0; d < 15; ++d)
            fSliders[d]->setVisible(!manual);
        fAudioInToggle->setVisible(!manual);
        fAutoPairToggle->setVisible(!manual);
        fSmoothingSlider->setVisible(!manual);

        // Il conteggio reale per agente resta applicato anche a vista chiusa (nessun danno,
        // updateManualExciterSpec/ResonatorSpec nascondono comunque gli slot oltre il
        // conteggio) -- qui serve solo mostrare/nascondere il gruppo intero.
        const uint32_t excCount = (fCurrentExciterSpec != nullptr)
            ? static_cast<uint32_t>(fCurrentExciterSpec->size()) : 0;
        const uint32_t resCount = (fCurrentResonatorSpec != nullptr)
            ? static_cast<uint32_t>(fCurrentResonatorSpec->size()) : 0;

        for (uint32_t s = 0; s < 9; ++s)
            fExciterParamSliders[s]->setVisible(manual && s < excCount);
        for (uint32_t s = 0; s < 10; ++s)
            fResonatorParamSliders[s]->setVisible(manual && s < resCount);

        repaint();
    }

    // agentSpecs() usa le stesse chiavi di kExciterEnum ("bow","blow",...) per gli
    // eccitatori -- verificato in ParamRanges.hpp contro exciters.py PARAM_RANGES.
    void updateManualExciterSpec()
    {
        const uint32_t idx = fExciterChoice->getIndex();
        const auto& specs = phimo::agentSpecs();
        const auto it = specs.find(fExciterLabels[idx]);
        fCurrentExciterSpec = (it != specs.end()) ? &it->second.params : nullptr;

        const uint32_t count = (fCurrentExciterSpec != nullptr)
            ? static_cast<uint32_t>(fCurrentExciterSpec->size()) : 0;

        for (uint32_t s = 0; s < 9; ++s)
        {
            const bool active = s < count;
            fExciterParamSliders[s]->setVisible(active && fModeToggle->isChecked());
            if (!active)
                continue;
            const phimo::ParamSpec& spec = (*fCurrentExciterSpec)[s];
            fExciterParamSliders[s]->setName(spec.name);
            fExciterParamSliders[s]->setRange(spec.lo, spec.hi, spec.logScale);
            fExciterParamSliders[s]->setValueQuiet(paramSpecDenormalize(fExciterParamRaw[s], spec));
        }
        updateDescriptorRanges(idx);
        repaint();
    }

    // 2026-09-25 (porting di gui.py _agent_range): gli slider descrittori della vista Agent
    // mostrano il range dell'ECCITATORE selezionato (kExciterDescriptorRanges,
    // ParamLayout.hpp), non il range host (unione di tutti). Il clamp vero dei valori
    // host al cambio di eccitatore lo fa il DSP (run(), syncDescriptorsToExciter), che
    // poi rimanda i valori qui via parameterChanged.
    void updateDescriptorRanges(uint32_t excIdx)
    {
        for (uint32_t d = 0; d < 15; ++d)
        {
            const DescriptorLoHi r = exciterDescriptorRange(static_cast<int>(excIdx), d);
            fSliders[d]->setRange(r.lo, r.hi, kDescriptorRanges[d].logScale);
            fSliders[d]->setValueQuiet(fDescriptorValue[d]);  // solo display, clamp nel widget
        }
    }

    // agentSpecs() usa le chiavi "resonator_<forma>" (resonator.py PARAM_RANGES) mentre
    // kResonatorEnum/fResonatorLabels ha solo "<forma>" -- serve il prefisso, verificato in
    // ParamRanges.hpp.
    void updateManualResonatorSpec()
    {
        const uint32_t idx = fResonatorChoice->getIndex();
        const std::string key = std::string("resonator_") + fResonatorLabels[idx];
        const auto& specs = phimo::agentSpecs();
        const auto it = specs.find(key);
        fCurrentResonatorSpec = (it != specs.end()) ? &it->second.params : nullptr;

        const uint32_t count = (fCurrentResonatorSpec != nullptr)
            ? static_cast<uint32_t>(fCurrentResonatorSpec->size()) : 0;

        for (uint32_t s = 0; s < 10; ++s)
        {
            const bool active = s < count;
            fResonatorParamSliders[s]->setVisible(active && fModeToggle->isChecked());
            if (!active)
                continue;
            const phimo::ParamSpec& spec = (*fCurrentResonatorSpec)[s];
            fResonatorParamSliders[s]->setName(spec.name);
            fResonatorParamSliders[s]->setRange(spec.lo, spec.hi, spec.logScale);
            fResonatorParamSliders[s]->setValueQuiet(paramSpecDenormalize(fResonatorParamRaw[s], spec));
        }
        repaint();
    }

    DISTRHO_DECLARE_NON_COPYABLE(MultiPhiMoUI)
};

UI* createUI()
{
    return new MultiPhiMoUI();
}

END_NAMESPACE_DISTRHO

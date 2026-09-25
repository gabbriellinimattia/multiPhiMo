#pragma once
// Widget custom per la GUI DGL di MultiPhiMo (punto 8A) -- disegno vettoriale via NanoVG
// (nessun asset bitmap, deciso con l'utente 2026-09-23), stesso pattern verificato in
// native/dpf/examples/FileHandling/NanoButton.{hpp,cpp} (NanoSubWidget + ButtonEventHandler,
// loadSharedResources() per il font "sans" incluso in DPF, nessun file esterno da fornire).
//
// Slider: drag manuale (onMouse/onMotion), nessun mixin EventHandler -- il valore reale
// (non normalizzato 0-1, stesso schema dei parametri host: parameter.ranges.min/max sono
// gia' il range fisico vero, vedi ParamLayout.hpp) e' mappato linearmente in log10 quando
// logScale=true, stesso schema di LOG_SCALE_KEYS/_slider_pos in gui.py.
//
// Toggle/CycleChoice: ButtonEventHandler (dgl/EventHandlers.hpp) con setCheckable(true) per
// Toggle -- verificato nel sorgente (dgl/src/EventHandlers.cpp) che il flip di isChecked()
// avviene GIA' internamente al completamento del click quando checkable=true, quindi
// buttonClicked() qui legge isChecked() gia' aggiornato, non lo flippa una seconda volta.
//
// Bug diagnosticato 2026-09-23 (contenuto che non scala col resize): NanoBaseWidget ha piu'
// overload di costruttore (NanoVG.hpp:924-934) scelti in base al tipo STATICO del parametro
// -- Widget* crea un contesto NanoVG separato per il widget (setNeedsViewportScaling, per
// sotto-finestre native), mentre NanoTopLevelWidget*/NanoSubWidget* CONDIVIDONO il contesto
// del genitore. I costruttori qui sotto prendevano tutti "Widget* const parent" -> finivano
// sempre nel primo overload, contesto isolato che non riceve lo scaling automatico del
// contesto principale. Tutti i widget sono figli diretti della UI top-level in questo round,
// quindi il parametro e' tipizzato NanoTopLevelWidget* per selezionare l'overload corretto.

#include "NanoVG.hpp"
#include "Color.hpp"
#include "EventHandlers.hpp"

#include <cmath>
#include <cstdio>
#include <functional>

START_NAMESPACE_DGL

// ---------------------------------------------------------------------------------

class Slider : public NanoSubWidget
{
public:
    explicit Slider(NanoTopLevelWidget* const parent)
        : NanoSubWidget(parent),
          fLo(0.0f), fHi(1.0f), fValue(0.0f), fLogScale(false), fDragging(false),
          fName("")
    {
        loadSharedResources();
    }

    void setRange(float lo, float hi, bool logScale) noexcept
    {
        fLo = lo; fHi = hi; fLogScale = logScale;
    }

    void setName(const char* name) noexcept { fName = name; }

    // Aggiorna il valore SENZA notificare onChanged -- usato da parameterChanged() per
    // riflettere lo stato dell'host senza innescare un loop di callback.
    void setValueQuiet(float value) noexcept
    {
        fValue = clampToRange(value);
        repaint();
    }

    float getValue() const noexcept { return fValue; }

    std::function<void(float)> onChanged;
    std::function<void()> onDragStart;
    std::function<void()> onDragEnd;

protected:
    void onNanoDisplay() override
    {
        const uint w = getWidth();
        const uint h = getHeight();
        const float trackY = 20.0f;
        const float trackH = h - trackY - 4.0f;

        // sfondo
        beginPath();
        fillColor(Color(40, 40, 44));
        rect(0, trackY, w, trackH);
        fill();
        closePath();

        // riempimento proporzionale al valore
        const float frac = valueToFraction(fValue);
        beginPath();
        fillColor(Color(90, 150, 200));
        rect(0, trackY, w * frac, trackH);
        fill();
        closePath();

        // bordo
        beginPath();
        strokeColor(Color(90, 90, 96));
        rect(0.5f, trackY + 0.5f, w - 1.0f, trackH - 1.0f);
        stroke();
        closePath();

        // nome (in alto a sinistra) e valore (in alto a destra)
        fontSize(12.0f);
        textAlign(ALIGN_LEFT | ALIGN_TOP);
        fillColor(Color(220, 220, 220));
        text(2, 0, fName, nullptr);

        char buf[32];
        formatValue(buf, sizeof(buf));
        textAlign(ALIGN_RIGHT | ALIGN_TOP);
        fillColor(Color(180, 200, 220));
        text(w - 2, 0, buf, nullptr);
    }

    bool onMouse(const MouseEvent& ev) override
    {
        if (ev.button != 1)
            return false;

        // WidgetPrivateData::giveMouseEventForSubWidgets (WidgetPrivateData.cpp) inoltra
        // l'evento a OGNI subwidget in ordine inverso finche' uno risponde true -- NON fa
        // hit-testing per posizione (a differenza di quanto assunto qui inizialmente):
        // sta al widget stesso verificare contains(ev.pos) prima di accettare il click,
        // stesso schema di ButtonEventHandler::PrivateData::mouseEvent (EventHandlers.cpp).
        // Bug diagnosticato 2026-09-23: senza questo check ogni click, ovunque, veniva
        // accettato dall'ultimo slider creato (pitch, ultimo in kColX/riga) perche' era il
        // primo interpellato nell'iterazione inversa.
        if (ev.press)
        {
            if (!contains(ev.pos))
                return false;
            fDragging = true;
            if (onDragStart)
                onDragStart();
            setValueFromX(static_cast<float>(ev.pos.getX()));
            return true;
        }

        if (fDragging)
        {
            fDragging = false;
            if (onDragEnd)
                onDragEnd();
            return true;
        }

        return false;
    }

    bool onMotion(const MotionEvent& ev) override
    {
        if (!fDragging)
            return false;

        setValueFromX(static_cast<float>(ev.pos.getX()));
        return true;
    }

private:
    float fLo, fHi, fValue;
    bool fLogScale;
    bool fDragging;
    const char* fName;

    float clampToRange(float v) const noexcept
    {
        if (v < fLo) return fLo;
        if (v > fHi) return fHi;
        return v;
    }

    float valueToFraction(float v) const noexcept
    {
        if (fLogScale)
        {
            const float lv = std::log10(std::max(v, 1e-6f));
            const float llo = std::log10(std::max(fLo, 1e-6f));
            const float lhi = std::log10(std::max(fHi, 1e-6f));
            return (lhi > llo) ? (lv - llo) / (lhi - llo) : 0.0f;
        }
        return (fHi > fLo) ? (v - fLo) / (fHi - fLo) : 0.0f;
    }

    float fractionToValue(float frac) const noexcept
    {
        frac = frac < 0.0f ? 0.0f : (frac > 1.0f ? 1.0f : frac);
        if (fLogScale)
        {
            const float llo = std::log10(std::max(fLo, 1e-6f));
            const float lhi = std::log10(std::max(fHi, 1e-6f));
            return std::pow(10.0f, llo + frac * (lhi - llo));
        }
        return fLo + frac * (fHi - fLo);
    }

    void setValueFromX(float x)
    {
        const float w = static_cast<float>(getWidth());
        const float frac = (w > 0.0f) ? (x / w) : 0.0f;
        fValue = clampToRange(fractionToValue(frac));
        repaint();
        if (onChanged)
            onChanged(fValue);
    }

    void formatValue(char* buf, size_t bufSize) const
    {
        if (fHi - fLo < 2.0f && fHi < 10.0f)
            std::snprintf(buf, bufSize, "%.4f", fValue);
        else if (fHi < 100.0f)
            std::snprintf(buf, bufSize, "%.3f", fValue);
        else
            std::snprintf(buf, bufSize, "%.1f", fValue);
    }

    DISTRHO_LEAK_DETECTOR(Slider)
};

// ---------------------------------------------------------------------------------

class Toggle : public NanoSubWidget,
               public ButtonEventHandler,
               public ButtonEventHandler::Callback
{
public:
    explicit Toggle(NanoTopLevelWidget* const parent)
        : NanoSubWidget(parent),
          ButtonEventHandler(this),
          fLabel("")
    {
        loadSharedResources();
        ButtonEventHandler::setCheckable(true);
        ButtonEventHandler::setCallback(this);
    }

    void setLabel(const char* label) noexcept { fLabel = label; }

    void setCheckedQuiet(bool checked) noexcept
    {
        ButtonEventHandler::setChecked(checked, false);
        repaint();
    }

    bool isChecked() const noexcept { return ButtonEventHandler::isChecked(); }

    std::function<void(bool)> onToggled;

protected:
    void onNanoDisplay() override
    {
        const uint w = getWidth();
        const uint h = getHeight();
        const bool checked = ButtonEventHandler::isChecked();

        beginPath();
        fillColor(checked ? Color(90, 150, 200) : Color(50, 50, 56));
        rect(0, 0, w, h);
        fill();
        strokeColor(Color(90, 90, 96));
        stroke();
        closePath();

        fontSize(13.0f);
        textAlign(ALIGN_CENTER | ALIGN_MIDDLE);
        fillColor(Color(240, 240, 240));
        text(w / 2.0f, h / 2.0f, fLabel, nullptr);
    }

    bool onMouse(const MouseEvent& ev) override { return ButtonEventHandler::mouseEvent(ev); }
    bool onMotion(const MotionEvent& ev) override { return ButtonEventHandler::motionEvent(ev); }

    void buttonClicked(SubWidget*, int) override
    {
        repaint();
        if (onToggled)
            onToggled(ButtonEventHandler::isChecked());
    }

private:
    const char* fLabel;

    DISTRHO_LEAK_DETECTOR(Toggle)
};

// ---------------------------------------------------------------------------------
// PlayButton (punto 8A round 3.1, 2026-09-23): pulsante momentaneo hold-to-sustain --
// press=note-on, release=note-off, stesso schema gia' validato dal vivo per le note MIDI
// reali in run() (PluginMultiPhiMo.cpp maschera lo status con 0xF0, canale ignorato).
// NON basato su ButtonEventHandler (che espone solo buttonClicked a fine-click, nessun
// evento separato di press/release) -- onMouse manuale con contains(ev.pos) sul press,
// stesso identico schema gia' verificato in Slider::onMouse sopra (bug diagnosticato
// 2026-09-23 su un widget precedente senza questo check).
class PlayButton : public NanoSubWidget
{
public:
    explicit PlayButton(NanoTopLevelWidget* const parent)
        : NanoSubWidget(parent),
          fLabel(""),
          fPressed(false)
    {
        loadSharedResources();
    }

    void setLabel(const char* label) noexcept { fLabel = label; }

    std::function<void()> onPress;
    std::function<void()> onRelease;

protected:
    void onNanoDisplay() override
    {
        const uint w = getWidth();
        const uint h = getHeight();

        beginPath();
        fillColor(fPressed ? Color(90, 180, 110) : Color(50, 50, 56));
        rect(0, 0, w, h);
        fill();
        strokeColor(Color(90, 90, 96));
        stroke();
        closePath();

        fontSize(13.0f);
        textAlign(ALIGN_CENTER | ALIGN_MIDDLE);
        fillColor(Color(240, 240, 240));
        text(w / 2.0f, h / 2.0f, fLabel, nullptr);
    }

    bool onMouse(const MouseEvent& ev) override
    {
        if (ev.button != 1)
            return false;

        if (ev.press)
        {
            if (!contains(ev.pos))
                return false;
            fPressed = true;
            repaint();
            if (onPress)
                onPress();
            return true;
        }

        // Release tracciato a prescindere dalla posizione corrente del mouse (stesso
        // schema di Slider::onMouse per fDragging) -- il rilascio puo' avvenire fuori
        // dai bounds dopo un trascinamento, deve comunque generare il note-off.
        if (fPressed)
        {
            fPressed = false;
            repaint();
            if (onRelease)
                onRelease();
            return true;
        }

        return false;
    }

private:
    const char* fLabel;
    bool fPressed;

    DISTRHO_LEAK_DETECTOR(PlayButton)
};

// ---------------------------------------------------------------------------------
// Lista a comparsa per CycleChoice (round 1, richiesta dall'utente dopo test dal vivo
// 2026-09-23 al posto del "click per avanzare" iniziale). CONDIVISA tra i selettori:
// creata una volta sola nella UI ed e' l'ULTIMO widget aggiunto, quindi il primo
// interpellato nell'iterazione inversa di giveMouseEventForSubWidgets (vedi nota in
// Slider::onMouse) -- la rende overlay sopra tutto il resto finche' e' visibile.
// Nascosta di default (setVisible(false)): WidgetPrivateData salta i subwidget non
// visibili sia nel disegno (displaySubWidgets) sia negli eventi mouse
// (giveMouseEventForSubWidgets), quindi da chiusa non intercetta nulla -- verificato in
// WidgetPrivateData.cpp. E' figlia diretta della UI top-level (non del CycleChoice che la
// apre) e posizionata in coordinate assolute, cosi' non e' vincolata al rettangolo del
// selettore che l'ha aperta.
class DropdownPopup : public NanoSubWidget
{
public:
    static constexpr uint kRowHeight = 24;

    explicit DropdownPopup(NanoTopLevelWidget* const parent)
        : NanoSubWidget(parent),
          fLabels(nullptr), fCount(0), fCurrentIndex(0)
    {
        loadSharedResources();
        setVisible(false);
    }

    void open(int x, int y, uint width, const char* const* labels, uint32_t count,
              uint32_t currentIndex, std::function<void(uint32_t)> onSelect)
    {
        fLabels = labels;
        fCount = count;
        fCurrentIndex = currentIndex;
        fOnSelect = onSelect;
        setAbsolutePos(x, y);
        setSize(width, count * kRowHeight);
        setVisible(true);
        repaint();
    }

    void close()
    {
        setVisible(false);
        repaint();
    }

protected:
    void onNanoDisplay() override
    {
        const uint w = getWidth();
        const float totalH = static_cast<float>(fCount * kRowHeight);

        beginPath();
        fillColor(Color(45, 45, 50));
        rect(0, 0, w, totalH);
        fill();
        strokeColor(Color(90, 90, 96));
        stroke();
        closePath();

        fontSize(13.0f);
        textAlign(ALIGN_LEFT | ALIGN_MIDDLE);

        for (uint32_t i = 0; i < fCount; ++i)
        {
            const float rowY = static_cast<float>(i * kRowHeight);

            if (i == fCurrentIndex)
            {
                beginPath();
                fillColor(Color(70, 110, 150));
                rect(0, rowY, w, kRowHeight);
                fill();
                closePath();
            }

            fillColor(Color(230, 230, 230));
            text(8, rowY + kRowHeight / 2.0f, fLabels[i], nullptr);
        }
    }

    bool onMouse(const MouseEvent& ev) override
    {
        if (ev.button != 1 || !ev.press)
            return false;

        // Ultimo widget aggiunto alla UI (vedi commento sopra la classe) -- riceve
        // l'evento PRIMA di tutti gli altri finche' e' visibile. Un click fuori dalla
        // propria area chiude il popup e consuma comunque il click, per non far scattare
        // anche il widget sottostante sullo stesso click.
        if (!contains(ev.pos))
        {
            close();
            return true;
        }

        const uint32_t row = static_cast<uint32_t>(ev.pos.getY() / kRowHeight);
        if (row < fCount && fOnSelect)
            fOnSelect(row);
        close();
        return true;
    }

private:
    const char* const* fLabels;
    uint32_t fCount;
    uint32_t fCurrentIndex;
    std::function<void(uint32_t)> fOnSelect;

    DISTRHO_LEAK_DETECTOR(DropdownPopup)
};

// ---------------------------------------------------------------------------------
// Selettore con dropdown (round 1b, 2026-09-23): mostra titolo + valore corrente, il
// click apre una DropdownPopup condivisa (vedi sopra) con l'elenco delle opzioni.

class CycleChoice : public NanoSubWidget,
                     public ButtonEventHandler,
                     public ButtonEventHandler::Callback
{
public:
    explicit CycleChoice(NanoTopLevelWidget* const parent)
        : NanoSubWidget(parent),
          ButtonEventHandler(this),
          fTitle(""), fLabels(nullptr), fCount(0), fIndex(0), fPopup(nullptr),
          fOpenUpward(false)
    {
        loadSharedResources();
        ButtonEventHandler::setCallback(this);
    }

    void setTitle(const char* title) noexcept { fTitle = title; }

    // Popup condiviso creato/posseduto dalla UI (vedi UIMultiPhiMo.cpp) -- non di
    // proprieta' di questo widget, nessuna gestione lifetime qui.
    void setPopup(DropdownPopup* popup) noexcept { fPopup = popup; }

    void setOptions(const char* const* labels, uint32_t count) noexcept
    {
        fLabels = labels;
        fCount = count;
    }

    void setIndexQuiet(uint32_t index) noexcept
    {
        fIndex = (fCount > 0) ? (index % fCount) : 0;
        repaint();
    }

    uint32_t getIndex() const noexcept { return fIndex; }

    // Round 3.3a (2026-09-23): il popup si apre di default SOTTO il widget (comportamento
    // originale, invariato per Exciter/Resonator in cima alla finestra). Un widget vicino
    // al bordo INFERIORE della finestra (es. il selettore Scala, y=428 su una finestra di
    // 460px) aprirebbe il popup fuori dall'area visibile -- bug osservato dal vivo
    // (dropdown "non funziona": in realta' si apriva sotto, fuori canvas). setOpenUpward(true)
    // fa aprire il popup SOPRA il widget invece che sotto.
    void setOpenUpward(bool upward) noexcept { fOpenUpward = upward; }

    std::function<void(uint32_t)> onChanged;

protected:
    void onNanoDisplay() override
    {
        const uint w = getWidth();
        const uint h = getHeight();

        beginPath();
        fillColor(Color(50, 50, 56));
        rect(0, 0, w, h);
        fill();
        strokeColor(Color(90, 90, 96));
        stroke();
        closePath();

        fontSize(11.0f);
        textAlign(ALIGN_CENTER | ALIGN_TOP);
        fillColor(Color(170, 170, 175));
        text(w / 2.0f, 2, fTitle, nullptr);

        fontSize(14.0f);
        textAlign(ALIGN_CENTER | ALIGN_MIDDLE);
        fillColor(Color(240, 240, 240));
        const char* label = (fLabels != nullptr && fCount > 0) ? fLabels[fIndex] : "";
        text(w / 2.0f, h / 2.0f + 6, label, nullptr);
    }

    bool onMouse(const MouseEvent& ev) override { return ButtonEventHandler::mouseEvent(ev); }
    bool onMotion(const MotionEvent& ev) override { return ButtonEventHandler::motionEvent(ev); }

    void buttonClicked(SubWidget*, int) override
    {
        if (fPopup == nullptr || fCount == 0 || fLabels == nullptr)
            return;

        const int popupY = fOpenUpward
            ? getAbsoluteY() - static_cast<int>(fCount * DropdownPopup::kRowHeight)
            : getAbsoluteY() + static_cast<int>(getHeight());

        fPopup->open(getAbsoluteX(), popupY,
                      getWidth(), fLabels, fCount, fIndex,
                      [this](uint32_t idx)
                      {
                          fIndex = idx;
                          repaint();
                          if (onChanged)
                              onChanged(fIndex);
                      });
    }

private:
    const char* fTitle;
    const char* const* fLabels;
    uint32_t fCount;
    uint32_t fIndex;
    DropdownPopup* fPopup;
    bool fOpenUpward;

    DISTRHO_LEAK_DETECTOR(CycleChoice)
};

// ---------------------------------------------------------------------------------

END_NAMESPACE_DGL

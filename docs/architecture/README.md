# Architecture diagram sources

Section 1.4 of the Software Architecture Document states:

> The diagram source is plain text and is held in the project repository under
> `docs/architecture/puml/`, so that every figure can be regenerated
> deterministically rather than redrawn by hand.

**That sentence is currently not true.** The sixteen `.puml` sources were never
committed. Until they are, nobody but their author can regenerate a figure,
which removes the whole reason for using a text-based diagramming tool.

## What to commit

One file per figure, named by content rather than by number, so that reordering
a section never invalidates a filename:

```
puml/
  _style.puml                  <- already here, shared by all sixteen
  01_view_model.puml
  02_architectural_style.puml
  03_system_context.puml
  04_use_case_view.puml
  05_logical_decomposition.puml
  06_classes_serving.puml
  07_classes_collection.puml
  08_activity_forecast.puml
  09_sequence_forecast.puml
  10_activity_collection.puml
  11_sequence_collection.puml
  12_deployment.puml
  13_components.puml
  14_source_packages.puml
  15_warehouse_schema.puml
  16_backup_flow.puml
```

## Two edits every source needs

```
@startuml
!include _style.puml          <- add this line
' title Figure 7 — ...        <- DELETE the title line entirely
```

The figure number belongs in the Word caption and nowhere else. It currently
lives in both places, the sections were reordered after the titles were
written, and five figures now print a number that contradicts their caption:

| Printed inside the image | Caption says |
|---|---|
| Figure 7 — Logical decomposition into subsystems | Figure 5 |
| Figure 5 — Architecturally significant classes: serving path | Figure 6 |
| Figure 6 — …classes: collection and training | Figure 7 |
| Figure 10 — Sequence: forecast request | Figure 9 |
| Figure 9 — Activity: one scheduled collection run | Figure 10 |

## Rendering

```bash
java -jar plantuml.jar -tpng -o ../png puml/*.puml
```

`_style.puml` sets `dpi 200` and a 28 px default font. See the comments in that
file for why: figures are placed in the document at a fixed 450 pt width, so a
2046 px-wide diagram has its labels printed at 3.1 pt. The five figures wider
than 1900 px need splitting or a landscape page as well as the font increase —
raising the font alone will not rescue them.

## Also outstanding

Prediction intervals are out of scope per SRS §1.2 and FR-09, so figures 5, 8
and 9 must lose the `IntervalEstimator` interface, its two implementations, the
`lower`/`upper` attributes on `HorizonForecast`, and the interval-derivation
step. See `Deliverables/ViewCastLK_SAD_v2.1_Correction_Pack.docx`, Part 3.

# Restyling toolbar icons into a measured visual language

Lessons from rebuilding a 34-icon Qt toolbar into the ParaView icon language, one
icon at a time, with the user reviewing each. Written as the operating procedure
for the next icon, not as a retrospective.

---

## 1. Measure the original before drawing anything

Every rejection in the first rounds came from drawing what I *thought* the icon
said. The originals turned out to be machine-built mosaics and their real content
differed from my impression in nearly every case.

Do this first, every time:

```
render the original at 10x with nearest-neighbour
dump a per-cell classification grid (letters: R/G/B/Y/M/K/D/m/w)
measure: text bbox, bar height, taper width, stroke width, every position
```

A real example: I "conserved" a lit sub-interval in a colour bar that did not
exist. The red cells were the ramp's own warm end, running
`#F58F0D → #EF5026 → #F52E2F → #E71D5A → #CC0E82` straight through. Sampling one
row proved it in seconds; three versions of the icon died before I checked.

**Rule: a design decision that costs a version of work should cost one measurement
first.**

## 2. Derive the target bands from the approved samples, by measurement

Do not eyeball "looks like ParaView". Render the reference icons, compute the
metrics, and use the range they actually span.

| Metric | Samples' range | Median | Notes |
|---|---|---|---|
| connected components | 1–5 | **1** | the "mosaic" complaint, made countable |
| ink density @64 px | 40–69 % | 55 % | below ~40 % reads as line art floating |
| tone layers (16-step luminance) | 3–10 | 8 | under 5 = flat line work |
| bbox span | 72–97 % | 92 % | how much of the canvas the ink fills |
| outline share | 3.8–54.8 % | — | **report only, never gate** |

`outline_share` is reported but not gated: measured across the samples it spans
3.8 % to 54.8 %, so any band would reject valid icons in both directions.

**These bands are diagnosis, not a verdict.** The user's eye decides. Reporting a
user-approved icon as "failing" because it misses a self-derived band inverts the
authority — the bands exist to find *what to look at*, not to overrule acceptance.

## 3. Know when a metric does not apply

Two cases where hitting the band would destroy the icon:

- **Point clouds fragment by definition.** Discrete dots cannot reach 55 % density
  without merging into a blob, and merging is exactly the "mosaic" defect. A cloud
  icon scoring 6–20 components and 20 % density is correct, not broken.
- **A Gaussian's filled area is only ~42 % of its own bounding box.** A bell curve
  cannot reach 55 % density without ceasing to look like a bell. Accept the low
  density and keep the identity.

Say so explicitly instead of silently bending the numbers or quietly failing.

## 4. Palette: measure frequency, do not invent

Sample every opaque pixel of the approved reference set and count colours. In one
measured set of 34,953 px the outline `#6F6F59` was 6.2 %, and **red-dominant
pixels were 0**. A whole icon had been built on brick red before anyone checked.

If a colour does not appear in the reference set, it is not in the language.
Compose the palette from measured stops, in an order the subject demands (cold to
warm for a scalar-field legend).

## 5. Text in the original is content — always keep it

The user's rule: **if the original icon contains text, the redraw must contain it
too, positioned sensibly.**

Short letter labels are how a user distinguishes six neighbouring filter buttons.
Dropping them is not a stylistic improvement; it is deleting information.

- Construct every letter as a path. Never `<text>` — it depends on the host having
  the font, and the host is a Qt application on an unknown machine.
- Distinguish *"the original had no meaningful graphic"* (free to redesign the
  graphic) from *"the original has no text"* (free to omit text). Do not confuse
  the two: an icon whose graphic is meaningless may still carry a necessary label.
- Check the original for text **before** designing, not after review.

### Letter geometry that survives 16 px

- Stroke weight dominates legibility far more than letterform. Measured case: the
  original's letters were 1 unit on a 21-unit canvas = 4.8 %; a redraw at 0.6 on 24
  = 2.5 % read as scratches no matter how carefully shaped. Match the *proportion*.
- Caps for labels at ≤5 units tall; lowercase counters close below that.
- Arch rise ≈ 23 % of x-height.
- A quadratic Bézier peaks at **half** its control offset. An arch with the control
  point at `y = top − rise` still reaches `y = top`, so place an overlying rule at
  `top − stroke/2 − clearance`, not at `top − rise`.
- A word's dot (`i`) goes **above** the rule it sits under, matching the source's
  own row order.

## 6. When text overlaps a mark, knock it out

If a label rides up into another element (a band, a ring, a bar), the underlying
dash or line runs through the letterforms and both become unreadable at 16 px.
Draw the same letter wider in the canvas colour first, then the ink letter on top.
This is depth ordering, not a cheat.

## 7. Compose, do not stack

A label sitting fully below a graphic leaves a gap. Measured case: a 3.40-unit gap
against a 12.70-unit cloud — almost exactly a quarter of the subject's height — and
the icon read as two stacked pieces rather than one mark.

**Overlap the label into roughly a quarter of the subject's height.** Measure the
overlap as a percentage of the subject's own extent, not as an absolute number.

The same logic fixed a "text too large" complaint. The cause was not the type size
but a **bar that had grown 31.7 % of the canvas against the source's 19.3 %**; the
bar squeezed the space and made correctly-sized letters look oversized. Fix the
structure, not the font.

## 8. Build one connected mass where the source does

Sample icon sets score `components = 1`. If the marks abut — rule touching letters,
letters touching a leader, leader touching a bar — the whole drawing is one mass.
Spacing each element with air around it is cleaner in isolation and reads as four
unrelated objects.

Measured case: chaining the original's own layout took an icon from 7 components to
2 without changing any element's design.

## 9. Reproduce semantics from the code, not from the name

Read the call site. Two toolbar buttons can share one function and differ only in a
parameter:

```cpp
if (filterType == BILATERAL) { /* compute a range sigma */ }
```

That single line is the whole difference between a Gaussian and a bilateral filter,
and it decides both icons: Gaussian smooths the edge away, bilateral preserves it. A
category-appropriate symbol chosen from the name alone would have made the two
buttons look the same.

Name the algorithm's actual output too: "gradient" turned out to be a *gradient
norm* (a scalar field of steepness), not a direction field.

## 10. Verify at the size the toolbar uses

Render 16 / 24 / 32 / 48 / 96 / 192 and look at the 16 px cells pixel-for-pixel.
An icon that works at 192 and fails at 16 is unfinished. Accept that dot detail
cannot survive 16 px (0.7 units ≈ 0.5 px) — make sure the *meaning-bearing* marks
do: text, the ✗, the band.

## 11. Working method

- **One icon at a time, with approval before the next.** A batch edit that touches
  several icons at once will be rejected even when each change is fine.
- **Show the comparison sheet inline** — original, new, and the approved samples,
  on one ladder. A file path is not a review.
- **Write the result to its real repository path** immediately. Never leave
  approved artwork in `/tmp`.
- **Name scripts for what they do**, not for the asset they replace. An `ecv`-style
  prefix inherited from the old resource tells the reader nothing about the tool.
- Keep drawing code free of CLI/rendering boilerplate: one module for the icons,
  one for previewing, scoring and installing them.

## 12. Failure modes seen, in order of cost

1. Substituting a generic shape for the original's actual content.
2. Adding a mark the source does not have (a "lit interval" that was the ramp).
3. Sizing type by eye instead of measuring the source's proportions.
4. Drawing a *consequence* of the algorithm instead of its *identity* (a smoothed
   step for a Gaussian).
5. Spacing elements apart where the source abuts them.
6. Leaving a gap where the composition wants an overlap.
7. Reporting self-derived metrics as if they had the user's authority.

## Analysis notes

CMS analysis notes are `analysis_note` records with ids `an:<note id>`
(for example `an:HIG-99-001` or the legacy `an:AN-99-001`). A note
contains `section` records (`an_section:<note id>:<file>`) and
`note_figure` records (`an_figure:<note id>:<label>`), and cites
`reference` records. Read a note's prose through its sections; a figure
has the authors' caption and label, not a description of the image. A
note links to the TWiki pages its bibliography names, and to its PAG.
Cite a note by its id and the section file.

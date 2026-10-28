# Priprema podataka za studiju Lijevče polje: vodič korak po korak

Ovaj vodič je namijenjen osobi koja priprema referentne podatke (tačke 1 do 3 ispod) i šalje ih za završnu
provjeru modela. Cilj je da se **svaki problem u podacima uhvati odmah pri prijemu**, uz jasnu poruku, a ne
tek nakon višesatnog treniranja. Provjeru radi naredba `geocrop intake`; ona ništa ne mijenja u vašim fajlovima.

Svi primjeri u ovom vodiču i u mapi [`examples/intake_example/`](https://github.com/ZeljkoDzafic/geocrop-engine/tree/main/examples/intake_example)
su **sintetički** (izmišljeni, ne potiču ni iz kakve evidencije).

## 1. Šta treba pripremiti

Tri sloja, svaki u zasebnom fajlu:

| # | Sloj | Šta je | Fajl (preporučeno) |
|---|---|---|---|
| 1 | **Trening poligoni** | referentni poligoni sa poznatom kulturom (skup od 366 poligona, samo oni unutar studijskog područja) | `training_2023.gpkg` |
| 2 | **Prijavljene parcele (LPIS/RPG) 2023** | parcele sa kulturom koju je proizvođač prijavio, uz šifre kultura iz šifarnika organa | `lpis_2023.gpkg` |
| 3 | **Nezavisna terenska opažanja** | tačke sa klasom i datumom, snimljene na terenu | `field_observations_2023.gpkg` |

Uz to, ako parcele imaju šifre kultura (a ne nazive klasa), potrebna je i **tabela šifara**
(`crop_mapping_rs.csv`, vidi dio 4.4).

Zašto tri odvojena sloja: trening poligoni služe za učenje modela, opažanja za **nezavisnu** provjeru tačnosti,
a parcele za provjeru prijava. Ista površina ne smije biti i u treningu i u provjeri (curenje podataka); `intake`
to provjerava.

## 2. Formati i koordinatni sistem

- **Preporučeni format:** GeoPackage (`.gpkg`), jedan sloj po fajlu. Prihvataju se i GeoJSON, Shapefile
  (obavezno sa `.shp`, `.shx`, `.dbf` i `.prj`), FlatGeobuf i GeoParquet.
- **Tabele (CSV)** za parcele i opažanja: vidi dio 6.
- **Koordinatni sistem (CRS) mora biti definisan.** Može ostati državni sistem u kojem su podaci nastali;
  program ga sam pretvara u EPSG:32633 (UTM zona 33N), koji studija koristi. Ako QGIS pokazuje „nema CRS“,
  sloj nije spreman (dio 5, greška „CRS nedostaje“).
- **Kodiranje znakova:** UTF-8 (š, č, ć, ž, đ ispravno). Za Shapefile/CSV iz starijih Windows programa često je
  cp1250; program to pokušava prepoznati, a možete ga i zadati (`--encoding cp1250`).
- Geometrija: trening poligoni i parcele su *Polygon/MultiPolygon*, opažanja su *Point*. Bez praznih i
  neispravnih geometrija.

## 3. Obavezna polja i primjeri tačnih tabela

Imena polja pišite tačno ovako (mala slova, bez razmaka).

### 3.1 Trening poligoni

| polje | tip | opis |
|---|---|---|
| `polygon_id` | tekst | **jedinstven** identifikator poligona, bez praznih vrijednosti |
| `class_name` | tekst | klasa: `soybean`, `sunflower`, `maize` (ciljne) i pomoćne klase (npr. `grassland`, `forest`, `winter_crops`) |
| `municipality` | tekst | *preporučeno:* `Gradiška`, `Laktaši` ili `Srbac` (omogućava brojanje po opštinama) |

```text
polygon_id | class_name | municipality
TSOY00     | soybean    | Gradiška
TSUN07     | sunflower  | Laktaši
TMAI03     | maize      | Srbac
TGRA11     | grassland  | Gradiška
```

Nazive ciljnih klasa pišite **engleski, malim slovima, bez razmaka** (`soybean`, ne „Soja “ ni „Soybean“):
program upoređuje tekst slovo po slovo. Raspon i dozvoljeni nazivi pomoćnih klasa dogovaraju se unaprijed.

### 3.2 Prijavljene parcele (LPIS/RPG)

| polje | tip | opis |
|---|---|---|
| `parcel_id` | tekst | **jedinstven** identifikator parcele (nemojte slati ime ili JMBG vlasnika) |
| `declared_crop` | tekst | prijavljena kultura: šifra iz šifarnika organa **ili** naziv klase; šifre čuvajte kao **tekst** (`0101`, ne `101`) |

```text
parcel_id | declared_crop
PSOY00    | SOJA
PSUN03    | SUNCOKRET
PMAI12    | KUKURUZ
```

**Ne šaljite** polja sa ličnim podacima (ime i prezime, adresa, JMBG, broj gazdinstva, telefon). Ako ih fajl ima,
obrišite ih prije slanja (dio 8).

### 3.3 Terenska opažanja

| polje | tip | opis |
|---|---|---|
| `class_name` | tekst | opažena kultura, istim nazivima kao trening (`soybean`, `sunflower`, `maize`, ...) |
| `observed_at` | datum | datum opažanja u **ISO formatu `GGGG-MM-DD`**, npr. `2023-07-14` |
| `group_id` | tekst | *preporučeno:* oznaka parcele/bloka; više tačaka na istoj parceli moraju imati istu oznaku (da se ne računaju kao nezavisne) |

```text
class_name | observed_at | group_id
soybean    | 2023-07-14  | PSOY00
maize      | 2023-07-15  | PMAI12
```

Datum **nije** `14.07.2023.` ni `14/07/2023` ni `jul 2023` (ovakve datume program odbija; vidi dio 6 za
automatsku pretvorbu). U polju `observed_at` u GeoPackage-u koristite tip „tekst“ ili „datum“.
Opažanja moraju biti iz sezone 2023 (proljeće do jesen, 15.03. do 31.10.2023.).

## 4. Šta `geocrop intake` provjerava

### 4.1 Za svaki sloj
Da se fajl može pročitati; da CRS postoji; neispravne geometrije (i koliko ih program može sam popraviti);
prazna i dupla polja ID; obavezna polja; pravopis naziva klasa (velika/mala slova, razmaci na kraju, dijakritici,
pogrešno dekodirani znakovi, uz **predloženu tablicu normalizacije**); koliko je elemenata unutar studijskog
područja (granica `aoi.geojson`); broj elemenata po opštinama (iz polja `municipality`/`opstina`; ako polja nema,
a nema ni fajla sa granicama opština, ta provjera se preskače uz napomenu).

### 4.2 Trening poligoni
Broj poligona, površina i broj piksela (10 m) po klasi; najmanji poligon u pikselima; poligoni manji od jednog
piksela; da li svaka klasa može učestvovati u prostornoj unakrsnoj validaciji (blokovi od 5 km, 5 slogova) i u
zaključanom izdvojenom skupu (provjera je indikativna sa 20 % ako izdvojeni skup nije podešen); upozorenje na
neuravnotežene klase (omjer najveće i najmanje klase veći od 10, empirijski prag); poligoni koji leže unutar
prijavljenih parcela (curenje pri provjeri prijava).

### 4.3 Opažanja
Format datuma, datumi van sezone studije, prazna polja grupe, duplikati (ista tačka, klasa i datum), klase koje
nema u treningu, te tačke u poligonu (ili unutar 50 m od poligona) za trening: takve se tačke ne smiju koristiti
za nezavisnu provjeru.

### 4.4 Parcele i tabela šifara
Šifre prijava koje **nisu u tabeli šifara** (greška ako je to 5 % ili više parcela), nepopunjene prijave, duple
geometrije, parcele premale za odluku nakon unutrašnjeg bafera od 1 piksela. Tabela šifara je CSV sa dvije kolone:

```text
source,class_name
SOJA,soybean
SUNCOKRET,sunflower
KUKURUZ,maize
PSENICA,winter_crops
```

Razdjelnik može biti `,` ili `;`, kodiranje UTF-8 ili cp1250. Svaka ciljna klasa mora imati bar jedan red.
Ista šifra ne smije imati dvije različite klase.

## 5. Česte greške i kako ih ispraviti u QGIS-u

| poruka u izvještaju | uzrok | ispravka u QGIS-u |
|---|---|---|
| `*.read` „Cannot read ...“ | fajl oštećen ili nedostaju dijelovi Shapefile-a | izvezite sloj ponovo: desni klik na sloj, *Izvezi, Sačuvaj osobine kao...*, format GeoPackage |
| `audit.*.vector.crs` | CRS nije definisan | desni klik, *Sloj, Postavi CRS sloja*; zatim izvezite u GeoPackage. Ne koristite „Reprojektuj“ ako CRS nije bio poznat |
| `*.aoi_overlap` (0 % unutar) | pogrešan CRS ili zamijenjene x/y | provjerite da se sloj poklapa sa podlogom (OpenStreetMap); ispravite CRS |
| `geometry.repairable` / `audit.*.geometry.valid` | neispravni poligoni (presijecanje samog sebe) | *Vektor, Alati geometrije, Popravi geometrije*, ili *Provjeri validnost* da ih vidite |
| `audit.training.schema.unique` | dupli `polygon_id` | otvorite tabelu atributa, sortirajte po `polygon_id`, ispravite ili generišite nove: kalkulator polja `$id` |
| `training.class_name.variants` / `.whitespace` | `Soybean`, `soybean `, `SOYBEAN` | *Kalkulator polja*, ažurirajte `class_name` izrazom `lower(trim("class_name"))`; dijakritike ispravite po predloženoj tablici |
| `training.class_name.target_spelling` | ciljna klasa napisana drugačije od `soybean`/`sunflower`/`maize` | ažurirajte izrazom, npr. `CASE WHEN lower(trim("class_name")) IN ('soja','soybean') THEN 'soybean' ELSE "class_name" END` |
| `*.encoding` („Å¡“ umjesto „š“) | fajl čitan u pogrešnom kodiranju | prilikom uvoza/izvoza odaberite UTF-8 (u QGIS-u: *Kodiranje* u dijalogu); za CSV `--encoding cp1250` |
| `parcels.unmapped_codes` | šifre kultura koje nisu u tabeli šifara | dopunite `crop_mapping_rs.csv` prema šifarniku organa |
| `parcels.declared_crop.float_codes` | Excel pretvorio šifru u broj (`1101.0`) | kolonu u CSV-u čuvajte kao tekst; program čita `1101.0` kao `1101` |
| `observations.date_format` | datum nije `GGGG-MM-DD` | *Kalkulator polja*: `format_date(to_date("datum",'dd.MM.yyyy'),'yyyy-MM-dd')`; ili `geocrop normalize-observations` (dio 6) |
| `observations.season` | datum van sezone | provjerite godinu i unos; opažanja van sezone se ne koriste |
| `observations.groups` | prazna oznaka grupe | popunite `group_id` (ista oznaka za tačke na istoj parceli) |
| `observations.training_overlap` | opažanje je u trening poligonu ili do 50 m od njega | uklonite tačku iz opažanja ili poligon iz treninga: ne smiju oba |
| `training.polygon_size` | poligoni manji od 1 piksela (10 m x 10 m) | provjerite poligone (npr. pogrešna jedinica ili greška crtanja) |
| `training.classes.blocks` / `fold_coverage` | klasa ima premalo prostorno razdvojenih poligona | prikupite dodatne poligone te klase na različitim lokacijama |

## 6. CSV/Excel tabele (WKT ili x/y) u neutralni GeoPackage

Ako parcele ili opažanja imate kao tabelu: sačuvajte iz Excela kao **CSV** (ne `.xlsx`; *Sačuvaj kao, CSV*,
razdjelnik `;` je u redu). Tabela mora imati geometriju na jedan od dva načina:

- kolona **WKT** (npr. `POLYGON((676000 4984000, ...))`) uz zadati CRS;
- kolone **x/y** ili **lon/lat** za tačke (zarez kao decimalni znak je dozvoljen).

```bash
# parcele iz CSV-a (WKT kolona), cp1250 i ; se prepoznaju sami; --encoding je dozvoljen eksplicitno
geocrop normalize-parcels --input parcele.csv --out lpis_2023.gpkg --id-field parcel_id \
    --declared-field declared_crop --crop-mapping crop_mapping_rs.csv --wkt-field wkt --crs EPSG:32633 --encoding cp1250

# opažanja iz CSV-a (x/y kolone); datumi dd.mm.gggg. se pretvaraju u ISO
geocrop normalize-observations --input opazanja.csv --out field_observations_2023.gpkg \
    --label-field klasa --date-field datum --group-field grupa --x-field x --y-field y --crs EPSG:32633
```

Datum koji nije ni ISO ni `dd.mm.gggg` program **ne pogađa**; odbija ga i navodi broj redova.
`geocrop intake` takođe može direktno čitati CSV (`--wkt-field`, `--x-field`, `--y-field`, `--crs`, `--encoding`,
`--delimiter`), ali je preporuka da prvo napravite GeoPackage naredbama iznad.

## 7. Pokretanje provjere

U radnom okruženju studije:

```bash
geocrop intake --config configs/studies/lijevce_polje/study_2023.yaml --require-all
```

Bez dodatnih opcija program uzima fajlove iz `paths:` u konfiguraciji (`data/lijevce_polje/reference/`). Fajlove
možete zadati i ručno: `--training`, `--parcels`, `--observations`, `--crop-mapping`. Rezultat je u
`intake_report.json` i `intake_report.md` (podrazumijevano u `outputs/lijevce_polje_2023/intake/`, ili `--out MAPA`).

- **PASS:** nema grešaka ni upozorenja.
- **PASS WITH WARNINGS:** nema grešaka; svako upozorenje treba pregledati i zabilježiti odluku.
- **FAIL:** postoji bar jedna greška; kod izlaza je `2`. Popravite podatke i pokrenite ponovo.

Opcija `--list-offenders N` ispisuje najviše N identifikatora (ili rednih brojeva redova) po neuspjeloj provjeri
(podrazumijevano 20, `0` isključuje). Preporuka je da dok popravljate podatke koristite taj ispis, a izvještaj
prije slanja pregledate (dio 8).

Isprobajte na sintetičkom primjeru: ispravan skup prolazi, a pokvareni pokazuje sve gore navedene greške.

```bash
geocrop intake --config examples/intake_example/study.yaml --out /tmp/intake_primjer
geocrop intake --config examples/intake_example/study.yaml --out /tmp/intake_primjer_greske \
    --training examples/intake_example/broken/training.geojson \
    --parcels examples/intake_example/broken/parcels.geojson \
    --observations examples/intake_example/broken/observations.geojson \
    --crop-mapping examples/intake_example/broken/crop_mapping.csv
```

## 8. Šta poslati, gdje to staviti i pravila privatnosti

**Šta poslati:** tri GeoPackage fajla (dio 1), tabelu šifara i **izvještaj `intake_report.md`** sa statusom PASS
ili PASS WITH WARNINGS (uz kratku napomenu o svakom upozorenju).

**Gdje staviti u radnom okruženju** (putanje iz `study_2023.yaml`):

```text
data/lijevce_polje/reference/training_2023.gpkg
data/lijevce_polje/reference/lpis_2023.gpkg
data/lijevce_polje/reference/field_observations_2023.gpkg
data/lijevce_polje/reference/crop_mapping_rs.csv
```

**Privatnost:**

1. Podaci o parcelama i opažanjima su zaštićeni i lični podaci; **nikada se ne objavljuju niti postavljaju u
   javni repozitorijum** (`data/` je isključen iz verzionisanja, ne zaobilazite to).
2. Šaljite samo polja iz dijela 3. Ime vlasnika, adresa, JMBG, broj gazdinstva, telefon: **ne**. Za identifikator
   parcele možete koristiti pseudonim (`geocrop normalize-parcels ... --pseudonymise` pravi nove ID-eve i tabelu
   veze koju **zadržite kod sebe**).
3. Izvještaj o prijemu sadrži samo brojeve, nazive klasa, šifre kultura i najviše N identifikatora; nikada
   geometrije, koordinate ni druge atribute. Ipak ga pregledajte prije dijeljenja; za izvještaj bez identifikatora
   koristite `--list-offenders 0`.
4. Fajlove šaljite zaštićenim kanalom dogovorenim sa timom, ne običnom e-poštom, i brišite privremene kopije.

## 9. Ograničenja provjere

- Prolaz provjere ne dokazuje da su oznake klasa tačne niti predviđa tačnost modela; govori samo da su podaci
  čitljivi, konzistentni i upotrebljivi za konfigurisani postupak.
- Pragovi označeni kao „empirijski“ (10 poligona po klasi, 25 opažanja po klasi, omjer klasa 10, 5 % neprepoznatih
  šifara) su radna pravila, nisu standard; prilagodite odluku stručnoj procjeni.
- Brojanje po opštinama bez polja `municipality` traži fajl sa granicama opština (`--municipalities`); tačne
  granice nisu dio ovog repozitorijuma.
- Prepoznavanje kodiranja je heuristika: ako se dijakritici pogrešno prikazuju, zadajte `--encoding`.
- Ovo je preporuka za ljudsku provjeru, ne konačan zaključak o kvalitetu podataka.

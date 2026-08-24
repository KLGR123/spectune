# RDKit (Python) 使用指南

处理分子结构、子结构搜索、指纹相似度、描述符、化学反应等任务时参考本指南。核心功能在 `rdkit.Chem`，3D构象生成/指纹生成器等进阶功能在 `rdkit.Chem.AllChem`。

## 读写分子
```python
from rdkit import Chem
m = Chem.MolFromSmiles('Cc1ccccc1')      # 也可 MolFromMolFile/MolFromMolBlock
```
失败返回 `None`，务必检查。批量读取用 Supplier，配合 `with` 并过滤 `None`：
```python
with Chem.SDMolSupplier('x.sdf') as suppl:
    mols = [m for m in suppl if m is not None]
```
`ForwardSDMolSupplier` 可读文件对象/gzip但不支持随机访问；`MultithreadedSDMolSupplier` 多线程读大文件。
写出：`Chem.MolToSmiles(m)`（规范化，`isomericSmiles=False`忽略立体；先`Kekulize`再传`kekuleSmiles=True`）、`Chem.MolToMolBlock(m)`；批量用`Chem.SDWriter('out.sdf')`。分子可用`pickle`或`m.ToBinary()`/`Chem.Mol(binStr)`序列化，比重新解析快很多，适合缓存。

## 分子操作
遍历：`m.GetAtoms()`、`m.GetBonds()`、`atom.GetNeighbors()`；按索引：`GetAtomWithIdx(i)`、`GetBondBetweenAtoms(i,j)`。环信息：`atom.IsInRing()`、`IsInRingSize(n)`。
加/去氢：`Chem.AddHs(m)`/`RemoveHs(m)`（生成3D构象前必须加氢）。2D坐标：`AllChem.Compute2DCoords(m)`。3D构象（ETKDG）：
```python
m3 = Chem.AddHs(m)
p = AllChem.ETKDGv3(); p.randomSeed = 42
AllChem.EmbedMolecule(m3, p)
```
> **谱图→结构推测场景**：加氢后统计每个碳/氮原子连接的氢数，可与 ¹H NMR 的积分/裂分（CH、CH₂、CH₃）逐一对照；环信息（`IsInRing`/`IsInRingSize`）用于核对谱图暗示的环状结构是否与候选分子一致。

## 绘图
```python
from rdkit.Chem import Draw
Draw.MolToFile(m, 'out.png')
Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200,200), legends=[...])
```
子结构对齐画图：`AllChem.GenerateDepictionMatching2DStructure(m, template)`。高亮匹配用`rdMolDraw2D.MolDraw2DCairo/SVG`+`PrepareAndDrawMolecule(d, mol, highlightAtoms=..., highlightBonds=...)`；可设`atom.SetProp('atomNote','foo')`标注，`d.drawOptions().addStereoAnnotation=True`显示R/S、E/Z。
> **谱图→结构推测场景**：辅助性工具——把候选 SMILES 画出来，并用`atomNote`标注推测的化学位移归属，方便与用户一起核对谱图-结构对应关系；本身不参与推理。

## 子结构搜索
```python
patt = Chem.MolFromSmarts('c[NH1]')   # SMARTS比SMILES语义更严格
m.HasSubstructMatch(patt)             # bool
m.GetSubstructMatches(patt)           # 所有匹配的原子索引元组
```
默认忽略手性，`useChirality=True`启用。SMARTS中`[C:1]`做原子映射，用`atom.GetAtomMapNum()`取回。高级过滤：`Chem.SubstructMatchParameters()` + `setExtraFinalCheck(callable)`自定义匹配后校验。
> **谱图→结构推测场景（重要）**：若从谱图特征（NMR化学位移范围、MS特征碎裂丢失）先推断出候选官能团（羰基、芳环、酯基等），可用SMARTS构造这些官能团的查询，去筛选/过滤候选分子库，快速排除不含该官能团的结构。

## 化学变换
删除/替换：`AllChem.DeleteSubstructs(m,patt)`、`AllChem.ReplaceSubstructs(m,patt,repl)`。SAR操作：`Chem.ReplaceSidechains(m,core)`（保留骨架标记侧链）、`Chem.ReplaceCore(m,core,labelByIndex=True)`（去骨架留侧链），配合`Chem.GetMolFrags(res,asMols=True)`拆分。Murcko骨架：
```python
from rdkit.Chem.Scaffolds import MurckoScaffold
core = MurckoScaffold.GetScaffoldForMol(m)
generic = MurckoScaffold.MakeScaffoldGeneric(core)
```

## 最大公共子结构
`rdFMCS.FindMCS([m1,m2,...])`返回单片段MCS（`.smartsString`）；`rdRascalMCES.RascalMCES`可返回多片段最大公共边子结构，更适合环差异大的分子对。
> **谱图→结构推测场景**：比较多个候选结构之间的共性骨架，或将候选结构与已知相似谱图对应的结构对齐，辅助判断哪个候选更合理。

## 指纹与相似度
推荐用指纹生成器接口，统一支持位/计数、折叠/稀疏形式：
```python
from rdkit import DataStructs
fpgen = AllChem.GetRDKitFPGenerator()          # 拓扑指纹
fpgen = AllChem.GetMorganGenerator(radius=2)   # Morgan/ECFP，radius*2≈直径
fpgen = AllChem.GetAtomPairGenerator()
fpgen = AllChem.GetTopologicalTorsionGenerator()
fp = fpgen.GetFingerprint(m)              # 位向量
fp = fpgen.GetSparseCountFingerprint(m)   # 稀疏计数向量
DataStructs.TanimotoSimilarity(fp1,fp2)   # 还有Dice/Cosine/Tversky等
```
Morgan传`atomInvariantsGenerator=AllChem.GetMorganFeatureAtomInvGen()`得类FCFP指纹。MACCS 166位指纹独立接口：`MACCSkeys.GenMACCSKeys(m)`。
> **谱图→结构推测场景（重要）**：若有已知谱-结构数据库，可先对未知谱图做粗略定性，再用Morgan/拓扑指纹做相似度检索，找到候选骨架作为推测起点；生成多个候选SMILES后，也可用指纹去重/聚类，避免同一结构的不同写法被当成不同候选。

## 描述符
```python
from rdkit.Chem import Descriptors
Descriptors.TPSA(m); Descriptors.MolLogP(m); Descriptors.MolWt(m)
vals = Descriptors.CalcMolDescriptors(m)   # 一次算全部约208个，返回dict，便于转DataFrame
```
Gasteiger电荷：`AllChem.ComputeGasteigerCharges(m)`后读`atom.GetDoubleProp('_GasteigerCharge')`；可用`SimilarityMaps.GetSimilarityMapFromWeights`可视化原子贡献。
> **谱图→结构推测场景（重要）**：MS方面，精确质量（`Descriptors.ExactMolWt`）、分子式、环+双键不饱和度（DBE）可用于从质荷比反推候选分子式，筛掉不合理候选；NMR方面，芳香性、杂原子计数、TPSA等描述符可辅助判断谱图对应的官能团大致范围。

## 化学反应
```python
rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OD1].[N!H0:3]>>[C:1](=[O:2])[N:3]')
products = rxn.RunReactants((mol1, mol2))  # 元组的元组：每组一套产物
```
也可从rxn文件读取。Recap（`rdkit.Chem.Recap`）、BRICS（`rdkit.Chem.BRICS`）用于逆合成式分子片段化。

## 类药性过滤
Lipinski五规则：分子量≤500、供体≤5（`NHOHCount`）、受体≤10（`NOCount`）、LogP≤5，违反不超1条视为通过。
不良子结构过滤（PAINS/Brenk/NIH）：
```python
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
p = FilterCatalogParams()
p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)  # 或 BRENK/NIH/ALL
cat = FilterCatalog(p)
cat.HasMatch(mol)      # bool
cat.GetMatches(mol)    # 详细条目，entry.GetDescription()
```

## 其他要点
- **Chem vs AllChem**：Chem为核心功能，AllChem额外含3D构象/指纹生成器等，常配合使用。
- 环信息非唯一（SSSR问题），用`Chem.GetSymmSSSR(m)`获取对称化环集。
- R基团分解：`rdkit.Chem.rdRGroupDecomposition`；超大虚拟库搜索：`rdkit.Chem.rdSynthonSpaceSearch`（支持子结构/相似度搜索）；药效团特征：`rdkit.Chem.rdMolChemicalFeatures`（配合`.fdef`定义文件）。
# Library

---

```@meta
CurrentModule = TaylorIntegration
```

## Exported functions

```@docs
taylorinteg
taylorinteg!
lyap_taylorinteg
@taylorize
firsttime
lasttime
```

## Exported types

```@docs
TaylorSolution
```

## Internal

```@docs
jetcoeffs!
```

```@autodocs
Modules = [TaylorIntegration]
Public = false
Filter = f -> f !== TaylorIntegration.jetcoeffs!
```

## Index

```@index
Pages = ["api.md"]
Order = [:function]
```

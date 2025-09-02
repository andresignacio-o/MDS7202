Trabajo en el curso de lab mds.
Javier Zapata y Andrés Oñate.

___ 

# Resumen Lab2


# Resumen: Sobrecarga de operadores en clases de Python

## 1. Objetos, atributos y métodos

* Una **clase** es un molde para crear **objetos**.
* Los **atributos** son los datos que guarda cada objeto.
* Los **métodos** son las funciones que saben usar esos datos.

Ejemplo:

```python
class Persona:
    def __init__(self, nombre, edad):
        self.nombre = nombre   # atributo
        self.edad = edad       # atributo
    
    def saludar(self):         # método
        return f"Hola, soy {self.nombre}"

p = Persona("Ana", 20)
print(p.saludar())   # Hola, soy Ana
```

Nota: El `__init__` es un método especial que se llama automáticamente cuando creas un nuevo objeto. Aquí es donde inicializas los atributos del objeto.

---

## 2. Métodos especiales (`__add__`, `__sub__`, …)

Python te deja decidir cómo deben comportarse los operadores (`+`, `-`, `*`, …) con tus objetos.

* `__add__(self, other)` → se usa cuando haces `obj + other`.
* `__radd__(self, other)` → se usa cuando haces `other + obj`.
* `__sub__`, `__rsub__`, etc. funcionan igual, pero con `-`.

👉 **self** = el objeto que está a la **izquierda** del operador.
👉 **other** = lo que está a la **derecha**.

---

## 3. ¿Por qué existen `radd`, `rsub`?

Cuando haces `5 + obj`, Python primero intenta `5.__add__(obj)`.

* Como `int` no sabe sumar tu objeto, responde *no sé*.
* Entonces Python intenta lo inverso: `obj.__radd__(5)`.

Si no tienes `__radd__` definido → error.

---

## 4. ¿Por qué con suma sí, y con resta no?

* La **suma** es conmutativa: `a + b = b + a`.

  * Por eso `__radd__` puede simplemente llamar a `__add__`.

* La **resta** NO es conmutativa: `a - b ≠ b - a`.

  * Por eso en `__rsub__` hay que escribir la operación al revés.

---

## 5. Ejemplos prácticos

### Ejemplo con números

```python
class Numero:
    def __init__(self, valor):
        self.valor = valor

    def __add__(self, other):
        return Numero(self.valor + other)

    def __radd__(self, other):
        return self.__add__(other)  # da igual el orden

    def __sub__(self, other):
        return Numero(self.valor - other)

    def __rsub__(self, other):
        return Numero(other - self.valor)  # orden invertido
```

Pendiente: Explicar bien que hace el return con self.__add__(other)

Pruebas:

```python
n = Numero(100)

print((n + 50).valor)   # 150
print((50 + n).valor)   # 150  (usa __radd__)

print((n - 50).valor)   # 50
print((50 - n).valor)   # -50  (usa __rsub__)
```

---

## 6. Llevado a imágenes

* `self.imagen` = el **array de pixeles** (ej. 600x400x3).
* `imagen + 50` = sumarle 50 a cada pixel (más brillo).
* `imagen - 50` = restarle 50 a cada pixel (más oscuro, pero nunca bajo 0).
* `50 - imagen` = lo contrario: cada pixel se convierte en `50 - valor`, y si es negativo se recorta a 0.

👉 En imágenes siempre recortamos los valores a `[0, 255]` porque son intensidades de color.

---

## 7. Ejemplo con un pixel de valor 100

* `imagen - 50` → `100 - 50 = 50`
* `50 - imagen` → `50 - 100 = -50`, pero saturamos → `0`
* `imagen + 200` → `100 + 200 = 300`, saturamos → `255`

---

✅ Así, cuando ves `__add__`, `__sub__`, `__radd__`, `__rsub__` en tu clase, no es algo misterioso:

* Solo son los **ganchos** que Python usa para decidir qué hacer con los operadores.
* Tú defines qué significa para tu tipo de objeto.


## 8. Parte 2.3

Se define la clase Imagen asi:

```python
class Imagen:
    def __init__(self, img):
        self.imagen = img
```

* Cuando se crea `gatito = Imagen(images["gatitos"][0])`,
  el arreglo NumPy que era la foto se guarda en el **atributo** `self.imagen`.




## 1. Lo que pasa al ejecutar

```python
gatito = Imagen(images["gatitos"][0])
```

* `images["gatitos"][0]` → es un **array NumPy** con los pixeles del primer gato.
* `Imagen(...)` → llama al constructor `__init__` de tu clase.
* Dentro del `__init__`, ese array se guarda en el atributo `self.imagen`.
* Como resultado, `gatito` se convierte en un **objeto de la clase `Imagen`** que envuelve esa foto.


---

## 2. Diferencia entre `img_in` y `img_in.imagen`

* `img_in` → es un **objeto de tu clase `Imagen`**.
  Es como una “cajita” que envuelve la foto.

* `img_in.imagen` → es el **arreglo NumPy real** que vive dentro de esa cajita.
  Es decir, los números `[R, G, B]` por pixel.

---

## 3. Ejemplo con números

Imagina una clase simple:

```python
class Numero:
    def __init__(self, valor):
        self.valor = valor
```

Si haces:

```python
n = Numero(5)
```

* `n` es un objeto `Numero`.
* `n.valor` es el número 5.

Del mismo modo:

* `img_in` es un objeto `Imagen`.
* `img_in.imagen` es el array de pixeles.

---

## 4. Entonces

Cuando dentro de tu método `to_gray` escribes:

```python
gris = np.dot(img_in.imagen, [0.299, 0.587, 0.114])
```

lo que haces es:
“Accede al array que está guardado dentro del objeto `img_in` y aplícale la fórmula del gris”.

---

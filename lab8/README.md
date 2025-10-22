docker build -t <nombre_imagen> .
    sirve para construir la imagen de docker con el nombre que se le pase despues del -t

docker run --rm -p 8000:80 <nombre_imagen>
    sirve para correr el contenedor de la imagen creada y mapear el puerto 80 del contenedor al puerto 8000 del host

Luego para abrir la app, abrir en el navegador la siguiente URL:
http://localhost:8000/docs
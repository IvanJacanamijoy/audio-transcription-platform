# Imagen base con PHP 8.0 CLI
FROM php:8.0-cli

# Instalar dependencias necesarias
RUN apt-get update && apt-get install -y \
    git \
    unzip \
    libpq-dev \
    libzip-dev \
    && docker-php-ext-install pdo pdo_pgsql zip

# Instalar Composer
COPY --from=composer:2 /usr/bin/composer /usr/bin/composer

# Directorio de la aplicación
WORKDIR /var/www/html

# Copiar código de Laravel
COPY . .

# Instalar dependencias de Laravel (sin ejecutar scripts de Artisan en el build)
RUN composer install --no-dev --optimize-autoloader --no-scripts

# Exponer puerto de Laravel
EXPOSE 8000

# Comando por defecto: levantar servidor con artisan
CMD ["php", "artisan", "serve", "--host=0.0.0.0", "--port=8000"]

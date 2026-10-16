# Carnívoros — Optimizador Inteligente de Cuentas por Pagar (USDT vs Bolívares)

Sistema de **Ciencia de Datos, Machine Learning y Simulación de Montecarlo** diseñado para optimizar las decisiones de tesorería y cuentas por pagar en Venezuela. Evalúa si realmente es conveniente aceptar ofertas de proveedores para liquidar facturas comerciales en **USDT (Cripto / Dólar Paralelo)** a cambio de un descuento nominal, o si es financieramente superior pagar la factura en **Bolívares (VES)** al vencimiento del crédito.

En el sector cárnico y comercial en Venezuela, es frecuente que los proveedores emitan facturas expresadas en Bolívares o indexadas a la tasa oficial del Banco Central de Venezuela (BCV), y posteriormente ofrezcan un "descuento" si la factura se cancela de inmediato en USDT:

* *Ejemplo típico:* Factura de **Bs. 150.000,00** (equivalente a **$170,43 USD** a tasa BCV de Bs. 880,12).
* El proveedor propone: *"Págame **160,00 USDT** hoy y te liquido la factura (te doy un 6% de descuento nominal)"*.

### La "Trampa" del Descuento Nominal vs. Real:
1. **Brecha Cambiaria (Spread):** En el mercado, 1 USDT en Binance P2P cotiza a **Bs. 1.002,00**, mientras que la tasa oficial BCV está en **Bs. 880,12** (una brecha del **13,85%**).
2. **Costo de Reposición en Bolívares:** Para entregar 160 USDT hoy, la empresa debe desembolsar o valorar:  
   $$160\text{ USDT} \times \text{Bs. } 1.002,00 = \text{Bs. } 160.320,00$$
3. **Resultado:** En lugar de ahorrar dinero, la empresa estaría pagando **Bs. 10.320,00 MÁS** que el monto original de la factura. El descuento nominal del 6% se convierte en una **pérdida real del -6,88%**.
4. **Horizonte de Crédito y Tendencia de la Brecha:** Si además la empresa tiene **15 o 30 días de crédito comercial**, pagar hoy en USDT implica renunciar a la liquidez y al rendimiento del capital en Bolívares. Si el modelo de Machine Learning proyecta que la brecha se va a expandir, pagar en USDT hoy es aún más perjudicial.

---


with last_order_date as (
    select max(order_date) as max_date from orders
), 
customer_stats as (
    select 
        c.customer_id,
        count(o.order_id) as total_orders,
        sum(od.unit_price * od.quantity) as total_spent,
        avg(od.unit_price * od.quantity) as avg_order_value
    from orders o
    inner join customers c on o.customer_id = c.customer_id
    inner join order_details od on o.order_id = od.order_id
    group by c.customer_id
),
label as ( 
    select 
        c.customer_id,
        case 
            when exists (
                select 1 
                from orders o2, last_order_date lod
                where o2.customer_id = c.customer_id 
                and o2.order_date >= (lod.max_date - interval '6 months')
            ) 
            then 1 else 0 
        end as will_order_again
    from customers c
)
select 
    s.customer_id,
    s.total_orders,
    s.total_spent,
    s.avg_order_value,
    l.will_order_again
from customer_stats s 
join label l on s.customer_id = l.customer_id;
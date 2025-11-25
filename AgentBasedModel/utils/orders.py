EPS = 1e-7


class Order:
    """
    Order contains all relevant information about order, it can be of two types: bid, ask. Supports binary comparison
    operations of price among all pairs of order types according to the logic:

    **better-offer < worse-offer**
    """
    order_id = 0

    def __init__(self, price, qty, order_type, trader_link=None):
        # Properties
        self.price = price
        self.qty = qty
        self.order_type = order_type
        self.trader = trader_link
        self.order_id = Order.order_id

        # Connections
        self.left = None
        self.right = None
        Order.order_id += 1

    def __lt__(self, other) -> bool:
        if self.order_type != other.order_type:
            if self.order_type == 'bid':
                return self.price > other.price
            if self.order_type == 'ask':
                return self.price < other.price

        if self.order_type == 'bid':
            return self.price > other.price
        if self.order_type == 'ask':
            return self.price < other.price

    def __le__(self, other) -> bool:
        if self.order_type != other.order_type:
            if self.order_type == 'bid':
                return self.price >= other.price
            if self.order_type == 'ask':
                return self.price <= other.price

        if self.order_type == 'bid':
            return self.price >= other.price
        if self.order_type == 'ask':
            return self.price <= other.price

    def __gt__(self, other) -> bool:
        if self.order_type != other.order_type:
            if self.order_type == 'bid':
                return self.price < other.price
            if self.order_type == 'ask':
                return self.price > other.price

        if self.order_type == 'bid':
            return self.price < other.price
        if self.order_type == 'ask':
            return self.price > other.price

    def __ge__(self, other) -> bool:
        if self.order_type != other.order_type:
            if self.order_type == 'bid':
                return self.price <= other.price
            if self.order_type == 'ask':
                return self.price >= other.price

        if self.order_type == 'bid':
            return self.price <= other.price
        if self.order_type == 'ask':
            return self.price >= other.price

    def __repr__(self) -> str:
        return f'{self.order_type} (price={self.price}, qty={self.qty})'

    def to_dict(self) -> dict:
        return {'price': self.price, 'qty': self.qty, 'order_type': self.order_type,
                'trader_link': self.trader}

    @classmethod
    def from_dict(cls, order_dict):
        return Order(order_dict['price'], order_dict['qty'], order_dict['order_type'], order_dict.get('trader_link'))


class OrderIter:
    """
    Iterator class for OrderList
    """
    def __init__(self, order_list):
        self.order = order_list.first

    def __next__(self) -> Order:
        if self.order:
            next_order = self.order
            self.order = self.order.right
            return next_order
        raise StopIteration


# noinspection DuplicatedCode
class OrderList:
    """
    OrderList is implemented as a doubly linked list. It preserves the same order type inside,
    all orders are sorted according to best-offer -> worst-offer dynamically.

    remove, append, push: complexity O(1)

    insert, fulfill (for large qty): complexity O(n)
    """
    def __init__(self, order_type: str):
        self.first = None
        self.last = None
        self.order_type = order_type

    def __iter__(self) -> OrderIter:
        return OrderIter(self)

    def __bool__(self):
        return self.first is not None and self.last is not None

    def __len__(self):
        n = 0
        for order in self:
            n += 1
        return n

    def to_list(self) -> list:
        return [order.to_dict() for order in self]

    def remove(self, order: Order):
        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')
        if order == self.first:
            self.first = order.right
        if order == self.last:
            self.last = order.left

        if order.left:
            order.left.right = order.right
        if order.right:
            order.right.left = order.left

        order.left = None
        order.right = None

    def append(self, order: Order):
        # If wrong order type to insert
        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

        if not self.first:
            self.first = order
            self.last = order
            return

        self.last.right = order
        order.left = self.last
        self.last = order

    def push(self, order: Order):
        """
        Insert order in the beginning

        :param order: Order
        :return: void
        """
        # If wrong order type to insert
        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

        if not self.first:
            self.first = order
            self.last = order
            return

        self.first.left = order
        order.right = self.first
        self.first = order

        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

    def insert(self, order: Order):
        """
        Inserts order preserving best-offer -> worst-offer

        :param order: Order
        :return: void
        """
        # If wrong order type to insert
        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

        # If empty
        if self.first is None:
            self.append(order)
            return

        # Insert order in the beginning
        if order <= self.first:
            order.right = self.first
            self.first.left = order
            self.first = order
            return

        # Insert order in the middle
        for val in self:
            if order <= val:
                # If only 1 order in self
                if self.first == self.last:
                    self.push(order)

                order.left = val.left
                order.right = val
                order.left.right = order
                order.right.left = order
                return

        # Insert to the end
        self.append(order)

    def fulfill(self, order: Order, t_cost: float) -> Order:
        if order.order_type == self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

        for val in self:
            if order.qty < EPS:
                break
            if val > order:
                break

            # Solve orders
            tmp_qty = min(order.qty, val.qty)  # Quantity traded currently
            tmp_price = val.price  # Price traded
            val.qty -= tmp_qty
            order.qty -= tmp_qty

            # Solve cash and assets
            if order.order_type == 'bid':
                if order.trader is not None:
                    order.trader.cash -= tmp_price * tmp_qty * (1 + t_cost)
                    order.trader.assets += tmp_qty
                if val.trader is not None:
                    val.trader.cash += tmp_price * tmp_qty * (1 - t_cost)
                    val.trader.assets -= tmp_qty

            if order.order_type == 'ask':
                if order.trader is not None:
                    order.trader.cash += tmp_price * tmp_qty * (1 - t_cost)
                    order.trader.assets -= tmp_qty
                if val.trader is not None:
                    val.trader.cash -= tmp_price * tmp_qty * (1 + t_cost)
                    val.trader.assets += tmp_qty

            # Clear remaining
            if val.qty < EPS:
                self.remove(val)

        return order

    @classmethod
    def from_list(cls, order_list, sort=False):
        order_list = [Order(order['price'], order['qty'], order['order_type'],
                            order.get('trader_link')) for order in order_list]
        order_list_obj = OrderList(order_list[0].order_type)
        if sort:
            for order in order_list:
                order_list_obj.insert(order)
        else:
            for order in order_list:
                order_list_obj.append(order)
        return order_list_obj

import unittest
import pycg.pycg as cg
import numpy as np


class TestHomogeneousCoordinate(unittest.TestCase):
    def setUp(self):
        self.coord = cg.HomogeneousCoordinate(3, 4, 5, 6)

    def test_getting_object_values(self):
        getters = [getattr(self.coord, getter) for getter in "xyzw"]
        for n, getter in enumerate(getters):
            self.assertEqual(self.coord[n], getter)

    def test_setting_object_values(self):
        """
        we should be able to update the coordinate entries by both array indexing and value
        :return:
        """
        # update by calling the setter method
        value_to_set = 0
        [setattr(self.coord, setter, value_to_set) for setter in "xyzw"]
        for n, getter in enumerate("xyzw"):
            self.assertEqual(getattr(self.coord, getter), value_to_set)
            self.assertEqual(self.coord[n], value_to_set)

        # update by directly calling the array index
        value_to_set = 1000
        for n in range(4):
            self.coord[n] = value_to_set

        for n, getter in enumerate("xyzw"):
            self.assertEqual(getattr(self.coord, getter), value_to_set)
            self.assertEqual(self.coord[n], value_to_set)


class TestPoint(unittest.TestCase):
    def setUp(self):
        self.coord = cg.Point(3, 4, 5)

    def test_fields_accessable(self):
        self.assertEqual(self.coord.w, 1)


class TestVector(unittest.TestCase):
    def setUp(self):
        self.coord = cg.Vector(3, 4, 5)

    def test_fields_accessable(self):
        self.assertEqual(self.coord.w, 0)


class TestWorldObject(unittest.TestCase):

    def test_object_creation(self):
        world_obj = cg.WorldObject()

        # the object should be centered at the origin facing the positive z-axis
        self.assertTrue(np.array_equal(world_obj.get_position(), cg.Point(0, 0, 0)))
        self.assertTrue(np.array_equal(world_obj.get_orientation(), cg.Vector(0, 0, 1)))

    def test_translation(self):
        my_obj = cg.WorldObject()

        # We should be able to move the object multiple times and the position will move but not direction
        move_vector = np.array((1, 2, -5))
        my_obj.move(*move_vector)  # move the object in space
        self.assertTrue(np.array_equal(my_obj.get_position(), cg.Point(*move_vector)))

        # reversing the move gets you back to the origin
        my_obj.move(*(-move_vector))  # move the object in space
        self.assertTrue(np.array_equal(my_obj.get_position(), cg.Point()))

        # individual move functions execute properly
        attr_names = ["move_"+direction for direction in "xyz"]
        move_attrs = [getattr(my_obj, attribute) for attribute in attr_names]
        movement = 3
        for n, fn_call in enumerate(move_attrs):
            fn_call(movement)
            self.assertEqual(my_obj.get_position()[n],movement)

        # Move calls should cascade
        my_obj = cg.WorldObject()

        movement = 3
        my_obj.move_x(movement).move_y(movement).move_z(movement)
        self.assertTrue(np.array_equal(my_obj.get_position(), cg.Point(movement,movement,movement)))

    def test_rotation(self):
        my_obj = cg.WorldObject()

        #rotation about the y-axis by 90 degree should change the direction vector to x
        my_obj.rotate_y(90, units="deg")
        self.assertTrue(np.allclose(my_obj.get_orientation(), cg.Vector(1., 0, 0)))

        # now rotation it about the z-axis 90 degree should have it point to positive y
        my_obj.rotate_z(90, units="deg")
        print(my_obj.get_orientation())
        self.assertTrue(np.allclose(my_obj.get_orientation(), cg.Vector(0, 1., 0)))

        #rotation 90 degree about the x-axis should reset it to positive z
        my_obj.rotate_x(90, units="deg")
        self.assertTrue(np.allclose(my_obj.get_orientation(), cg.Vector(0, 0, 1.)))



if __name__ == '__main__':
    unittest.main()

from sys import arg

from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx, barrier
from gpu.memory import AddressSpace
from sys import has_accelerator
from random import random_ui64, seed
from layout import Layout, LayoutTensor
from memory import UnsafePointer, stack_allocation
from testing import assert_equal

alias int_type = DType.uint64
alias block_check_type = DType.uint32
alias BLOCK_DIM = 1024
# alias layout = Layout.row_major(vector_size)

fn prefix_sum(
    vector_size: Int,
    x: UnsafePointer[Scalar[int_type]],
    block_check: UnsafePointer[Scalar[block_check_type]],
):
    shared = stack_allocation[
        2 * BLOCK_DIM,
        Scalar[int_type],
        address_space = AddressSpace.SHARED,
    ]()

    var bid = block_idx.x
    var lid = thread_idx.x
    var gid = 2 * BLOCK_DIM * bid + lid
    if gid < vector_size:
        shared[lid] = x[gid]
    else:
        shared[lid] = 0
    if gid + BLOCK_DIM < vector_size:
        shared[lid + BLOCK_DIM] = x[gid + BLOCK_DIM]
    else:
        shared[lid + BLOCK_DIM] = 0

    stride = 1
    while stride < 2 * BLOCK_DIM:
        # stride = 1, lid = 0: 2 * 1 * (0 + 1) - 1 =  1
        # stride = 2, lid = 0: 2 * 2 * (0 + 1) - 1 =  3
        # stride = 4, lid = 0: 2 * 4 * (0 + 1) - 1 =  7
        # stride = 8, lid = 0: 2 * 8 * (0 + 1) - 1 = 15
        # ...
        # stride = 1, lid = 1: 2 * 1 * (1 + 1) - 1 =  3
        # stride = 2, lid = 1: 2 * 2 * (1 + 1) - 1 =  7
        # stride = 4, lid = 1: 2 * 4 * (1 + 1) - 1 = 15
        # stride = 8, lid = 1: 2 * 8 * (1 + 1) - 1 = 31
        # ...
        # stride = 1, lid = 2: 2 * 1 * (2 + 1) - 1 =  5
        # stride = 2, lid = 2: 2 * 2 * (2 + 1) - 1 = 11
        # stride = 4, lid = 2: 2 * 4 * (2 + 1) - 1 = 23
        # stride = 8, lid = 2: 2 * 8 * (2 + 1) - 1 = 47
        # ...
        # stride = 1, lid = 256: 2 * 1 * (256 + 1) - 1 = 513
        # stride = 2, lid = 0: 2 * 2 * (0 + 1) - 1 =  1027
        # stride = 4, lid = 0: 2 * 4 * (0 + 1) - 1 =  2055 (done)
        # ...
        barrier()
        index = 2 * stride * (lid + 1) - 1
        if index < 2 * BLOCK_DIM:
            shared[index] += shared[index - stride]
        stride *= 2

    stride = (2 * BLOCK_DIM) // 4
    while stride > 0:
        barrier()
        index = (2 * (lid + 1) + 1) * stride - 1
        if index < 2 * BLOCK_DIM:
            shared[index] += shared[index - stride]
        stride //= 2
    prefix_sum = 0
    # TODO: handle the case of multiple blocks using block_check
    barrier()
    if gid < vector_size:
        x[gid] = shared[lid] + prefix_sum
    if gid + BLOCK_DIM < vector_size:
        x[gid + BLOCK_DIM] = shared[lid + BLOCK_DIM] + prefix_sum


def main():

    args = arg.argv()
    vector_size = 1000
    if len(args) > 1:
        vector_size = Int(args[1])

    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        print("Found GPU:", ctx.name())

        h_x = ctx.enqueue_create_host_buffer[int_type](
            vector_size
        )
        h_y = ctx.enqueue_create_host_buffer[int_type](
            vector_size
        )
        ctx.synchronize()

        seed(88)
        for i in range(vector_size):
            h_x[i] = random_ui64(0, 255).cast[int_type]()

        d_x = ctx.enqueue_create_buffer[int_type](vector_size)
        ctx.enqueue_copy(dst_buf=d_x, src_buf=h_x)

        num_blocks = (vector_size + 2 * BLOCK_DIM - 1) // (2*BLOCK_DIM)
        grid_dim = BLOCK_DIM * num_blocks
        d_block_check = ctx.enqueue_create_buffer[block_check_type](num_blocks).enqueue_fill(0)

        ctx.enqueue_function[prefix_sum](
            vector_size,
            d_x.unsafe_ptr(),
            d_block_check.unsafe_ptr(),
            grid_dim=grid_dim,
            block_dim=BLOCK_DIM,
        )

        ctx.enqueue_copy(dst_buf=h_y, src_buf=d_x)
        ctx.synchronize()

        sum = 0 * h_x[0]
        for i in range(vector_size):
            sum += h_x[i]
            assert_equal(sum, h_y[i])

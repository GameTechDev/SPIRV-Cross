///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2017, Intel Corporation
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Copyright 2015-2017 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SPIRV_CROSS_ISPC_HPP
#define SPIRV_CROSS_ISPC_HPP

#include "spirv_glsl.hpp"
#include <set>
#include <utility>
#include <vector>

namespace spirv_cross
{
class CompilerISPC : public CompilerGLSL
{
public:
	CompilerISPC(std::vector<uint32_t> spirv_)
	    : CompilerGLSL(move(spirv_))
	{
	}

	CompilerISPC(const uint32_t *ir, size_t word_count)
	    : CompilerGLSL(ir, word_count)
	{
	}

	std::string compile() override;

	// Sets a custom symbol name that can override
	// spirv_cross_get_interface.
	//
	// Useful when several shader interfaces are linked
	// statically into the same binary.
	void set_interface_name(std::string name)
	{
		interface_name = std::move(name);
	}

	// Sets the debug flags.
	// Currently disables inline functions
	void set_debug()
	{
		debug = true;
	}

	// Doesn't emit any runtime array padding.
	// HLSL doesn't seem to want to honour the arraystride option in SPV.
	// Could be a bug in the SPV translator?
	void set_ignore_runtimearray_padding()
	{
		ignore_runtimearray_padding = true;
	}

	// Group Syncs and Group Barriers are not currently supported by ISPC, so ignore them
	void set_ignore_group_barriers()
	{
		ignore_group_barriers = true;
	}

protected:
	// Specifically for the ISPC compiler
	// Look for potential targets to vectorise
	struct VectorisationHandler : OpcodeHandler
	{
		VectorisationHandler(CompilerISPC &compiler_)
		    : compiler(compiler_)
		{
		}

		void set_current_block(const SPIRBlock &) override;
		void set_current_function(const SPIRFunction &) override;
		bool begin_function_scope(const uint32_t *, uint32_t) override;
		bool end_function_scope(const uint32_t *, uint32_t) override;

		bool handle(spv::Op opcode, const uint32_t *args, uint32_t length) override;
		bool propogate_ispc_varyings_for_builtins();
		bool propogate_ispc_varyings(const uint32_t var);
		bool reverse_propogate_ispc_varyings(const uint32_t var);
		void dump_varying_dependancies();
		void dump_varying_dependancy_branch(const uint32_t index, uint32_t tab_count, bool recurse,
		                                    std::unordered_set<uint32_t> &branch);

		CompilerISPC &compiler;

		std::unordered_map<uint32_t, std::unordered_set<uint32_t>>
		    dependee_hierarchy; // a = b; a is dependant upon b, so [b] = a. If a is a varying, then b must be
		std::unordered_map<uint32_t, std::unordered_set<uint32_t>>
		    dependant_hierarchy; // a = b; a is dependant upon b, but for return types we propogate backwards so if a is a varying, the b should be
		// Used for reverse propogating varyings for return types. ie. If a func arg is a varying, then the return type is a varying.
		// Anything also assigned to the return type should be a varying
		std::set<uint32_t> reverse_propogation_varyings;
		const SPIRBlock *current_block = nullptr;
		const SPIRFunction *current_func = nullptr;
		std::stack<CFG *> cfg_stack;

		struct conditional_block_tracker
		{
			uint32_t condition = 0;
			uint32_t next_block = 0;
			uint32_t this_block = 0;
			const SPIRFunction *func = nullptr;
		};
		std::vector<conditional_block_tracker *> condition_block_stack;
	};

private:
	void emit_header() override;
	void emit_ispc_main();
	void emit_function_prototype(SPIRFunction &func, const Bitset &return_flags) override;

	void emit_resources();
	void emit_buffer_block(const SPIRVariable &type) override;
	void emit_push_constant_block(const SPIRVariable &var) override;
	void emit_interface_block(const SPIRVariable &type);
	void emit_block_chain(SPIRBlock &block);
	void emit_uniform(const SPIRVariable &var) override;
	void emit_shared(const SPIRVariable &var);
	void emit_block_struct(SPIRType &type);
	void emit_instruction(const Instruction &instruction) override;
	void emit_glsl_op(uint32_t result_type, uint32_t result_id, uint32_t op, const uint32_t *args,
	                  uint32_t count) override;
	std::string read_access_chain(const SPIRAccessChain &chain);
	void emit_load(const Instruction &instruction);
	void emit_store(const Instruction &instruction);
	void emit_access_chain(const Instruction &instruction);
	void emit_stdlib();
	void emit_struct(SPIRType &type);
	void emit_specialization_constants();
	void emit_workgroup_variables();

	bool maybe_emit_array_assignment(uint32_t id_lhs, uint32_t id_rhs);
    bool check_scalar_atomic_image(uint32_t id);

	std::string variable_decl(const SPIRType &type, const std::string &name, uint32_t id) override;

    std::string to_qualifiers_glsl(uint32_t id) override;
    std::string to_varying_qualifiers_ispc(uint32_t id) override;
    std::string argument_decl(const SPIRFunction::Parameter &arg) override;
	std::string type_to_glsl(const SPIRType &type, uint32_t id = 0) override;
	std::string type_to_glsl_constructor(const SPIRType &type) override;
	std::string bitcast_glsl_op(const SPIRType &result_type, const SPIRType &argument_type) override;
	bool optimize_read_modify_write(const SPIRType &type, const std::string &lhs, const std::string &rhs) override;

	void find_vectorisation_variables();

	void extract_global_variables_from_functions();
	std::unordered_map<uint32_t, std::set<uint32_t>> function_global_vars;
	void extract_global_variables_from_function(uint32_t func_id, std::set<uint32_t> &added_arg_ids,
	                                            std::unordered_set<uint32_t> &global_var_ids,
	                                            std::unordered_set<uint32_t> &processed_func_ids);

	void find_loop_variables_from_functions();
	void find_loop_variables_from_function(uint32_t func_id, std::unordered_set<uint32_t> &processed_functions);

	void localize_global_variables();
	std::string ensure_valid_name(std::string name, std::string pfx);
	std::string entry_point_args(bool append_comma, bool want_builtins, bool want_workgroup_vars);
	std::string entry_point_args_init(bool append_comma);
	void find_entry_point_args();
	std::string layout_for_member(const SPIRType &type, uint32_t index) override;
	bool optimize_read_modify_write(const std::string &, const std::string &)
	{
		return false;
	}

	std::vector<std::string> resource_registrations;
	std::vector<std::string> resource_entry_arguments;
	std::vector<std::string> resource_entry_arguments_init;

	std::vector<Variant *> entry_point_ids;
	std::vector<Variant *> workgroup_variable_ids;

	std::string impl_type;
	std::string resource_type;
	uint32_t shared_counter = 0;

	std::string interface_name;

	bool debug = false;
	bool ignore_runtimearray_padding = false;
	bool ignore_group_barriers = false;

	// stdlib codegen
	void codegen_constructor(std::string type, bool varying, uint32_t width, uint32_t arg_count, uint32_t arg_width[4]);
	void codegen_cast_constructor(std::string src_type, std::string dst_type, bool varying, uint32_t width);
    void codegen_matrix_constructor(std::string type, bool varying);
	void codegen_load_op(std::string type, uint32_t width);
	void codegen_store_op(std::string type, uint32_t width);
	void codegen_default_structs(std::string type, uint32_t width);
	void codegen_default_image_structs(uint32_t width);
	void codegen_default_pixel_structs(uint32_t width);
	void codegen_default_binary_op(std::string type, uint32_t width, std::string op);

	void codegen_unary_float_op(
	    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<std::uint32_t> vector_width,
	    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func);
	void codegen_unary_float_op_scalar_return(
	    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<std::uint32_t> vector_width,
	    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func);
	void codegen_unary_float_op_scalar_bool_return(
	    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<uint32_t> vector_width,
	    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func);
	void codegen_unary_float_op_simple(std::string func_name, std::vector<std::vector<std::string>> &varyings,
	                                   std::vector<uint32_t> vector_width);
	void codegen_unary_float_op_int_return(
	    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<uint32_t> vector_width,
	    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func);

	void codegen_binary_float_op(
	    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<std::uint32_t> vector_width,
	    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func);
	void codegen_binary_float_op_scalar_return(
	    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<std::uint32_t> vector_width,
	    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func);
	void codegen_binary_float_op_simple(std::string func_name, std::vector<std::vector<std::string>> &varyings,
	                                    std::vector<uint32_t> vector_width);

	void codegen_ternary_float_op(
	    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<std::uint32_t> vector_width,
	    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func);
	void codegen_ternary_float_op_scalar_return(
	    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<std::uint32_t> vector_width,
	    const std::function<void(std::vector<std::string> varyings, uint32_t vector_width)> &func);
	void codegen_ternary_float_op_simple(std::string func_name, std::vector<std::vector<std::string>> &varyings,
	                                     std::vector<uint32_t> vector_width);
	void codegen_ternary_float_op_multiple_widths(
	    std::string func_name, std::vector<std::vector<std::string>> &varyings,
	    std::vector<std::vector<uint32_t>> vector_widths,
	    const std::function<void(std::vector<std::string> varyings, std::vector<uint32_t> vector_widths)> &func);

	void codegen_unary_op_scalar_return(
	    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<std::string> &types,
	    std::vector<uint32_t> vector_width,
	    const std::function<void(std::vector<std::string> varyings, std::vector<std::string> types,
	                             uint32_t vector_width)> &func);

	void codegen_binary_op_scalar_return(
	    std::string func_name, std::vector<std::vector<std::string>> &varyings, std::vector<std::string> &types,
	    std::vector<uint32_t> vector_width,
	    const std::function<void(std::vector<std::string> varyings, std::vector<std::string> types,
	                             uint32_t vector_width)> &func);

	void codegen_binary_op(std::string func_name, std::vector<std::vector<std::string>> &varyings,
	                       std::vector<std::string> &types, std::vector<uint32_t> vector_width,
	                       const std::function<void(std::vector<std::string> varyings, std::vector<std::string> types,
	                                                uint32_t vector_width)> &func);

    void codegen_matrix_multiply(uint32_t dim, std::vector<std::vector<std::string>> &varyings);
    void codegen_matrix_transpose(uint32_t dim, std::vector<std::vector<std::string>> &varyings);
    void codegen_matrix_times_scalar(uint32_t dim, std::vector<std::vector<std::string>> &varyings);
    void codegen_vector_times_matrix(uint32_t dim, std::vector<std::vector<std::string>> &varyings);
    void codegen_matrix_times_vector(uint32_t dim, std::vector<std::vector<std::string>> &varyings);
    void codegen_matrix_inverse(std::string &varying);


};
} // namespace spirv_cross

#endif
